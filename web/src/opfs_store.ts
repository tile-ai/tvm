/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

interface OPFSWritableFileStream extends WritableStream<Uint8Array> {
  write(data: Blob | BufferSource | string): Promise<void>;
  close(): Promise<void>;
}

interface OPFSFileHandle {
  getFile(): Promise<Blob>;
  createWritable(): Promise<OPFSWritableFileStream>;
}

interface OPFSDirectoryHandle {
  getDirectoryHandle(
    name: string,
    options?: { create?: boolean },
  ): Promise<OPFSDirectoryHandle>;
  getFileHandle(
    name: string,
    options?: { create?: boolean },
  ): Promise<OPFSFileHandle>;
  removeEntry(name: string): Promise<void>;
}

interface OPFSStorageManager {
  getDirectory?: () => Promise<OPFSDirectoryHandle>;
}

interface OPFSStoreMetadata {
  url: string;
  contentType?: string;
}

const HASH_ALGORITHM = "SHA-256";
const OPFS_STORE_ROOT_DIRECTORY = "tvmjs-opfs-store";

export class OPFSStore {
  private readonly scope: string;
  private directoryPromise?: Promise<OPFSDirectoryHandle>;

  constructor(scope: string) {
    this.scope = scope;
  }

  static isAvailable(): boolean {
    const storage = OPFSStore.getStorageManager();
    return storage !== undefined && typeof storage.getDirectory === "function";
  }

  async has(url: string): Promise<boolean> {
    return (await this.read(url)) !== undefined;
  }

  async read(url: string): Promise<Response | undefined> {
    const directory = await this.getScopedDirectory();
    const baseName = await this.hashUrl(url);
    const dataHandle = await this.getFileHandleIfExists(
      directory,
      `${baseName}.bin`,
      false,
    );
    if (dataHandle === undefined) {
      return undefined;
    }
    const dataBlob = await dataHandle.getFile();
    const metadataHandle = await this.getFileHandleIfExists(
      directory,
      `${baseName}.meta.json`,
      false,
    );
    let metadata: OPFSStoreMetadata | undefined = undefined;
    if (metadataHandle !== undefined) {
      metadata = await this.readMetadata(metadataHandle);
      if (metadata?.url !== undefined && metadata.url !== url) {
        throw new Error("OPFSStore: metadata URL does not match key URL.");
      }
    }
    const headers =
      metadata?.contentType !== undefined
        ? { "content-type": metadata.contentType }
        : undefined;
    return new Response(dataBlob, headers ? { headers } : undefined);
  }

  async write(url: string, response: Response): Promise<void> {
    const directory = await this.getScopedDirectory();
    const baseName = await this.hashUrl(url);
    const dataHandle = await directory.getFileHandle(`${baseName}.bin`, {
      create: true,
    });
    const metadataHandle = await directory.getFileHandle(
      `${baseName}.meta.json`,
      { create: true },
    );
    const metadata: OPFSStoreMetadata = {
      url,
      contentType: response.headers.get("content-type") ?? undefined,
    };
    const writable = await dataHandle.createWritable();
    if (response.body !== null) {
      await response.body.pipeTo(writable);
    } else {
      await writable.write(await response.arrayBuffer());
      await writable.close();
    }
    await this.writeFile(
      metadataHandle,
      new TextEncoder().encode(JSON.stringify(metadata)),
    );
  }

  async remove(url: string): Promise<void> {
    const directory = await this.getScopedDirectory();
    const baseName = await this.hashUrl(url);
    await this.removeEntryIfExists(directory, `${baseName}.bin`);
    await this.removeEntryIfExists(directory, `${baseName}.meta.json`);
  }

  private static getStorageManager(): OPFSStorageManager | undefined {
    if (typeof navigator === "undefined") {
      return undefined;
    }
    return navigator.storage as unknown as OPFSStorageManager;
  }

  private async getScopedDirectory(): Promise<OPFSDirectoryHandle> {
    if (this.directoryPromise !== undefined) {
      return this.directoryPromise;
    }
    // Cache scoped directory handle to avoid repeated tree traversal
    this.directoryPromise = (async () => {
      const storage = OPFSStore.getStorageManager();
      if (storage === undefined || typeof storage.getDirectory !== "function") {
        throw new Error("OPFSStore: OPFS API unavailable.");
      }
      let directory = await storage.getDirectory();
      directory = await directory.getDirectoryHandle(OPFS_STORE_ROOT_DIRECTORY, {
        create: true,
      });
      const scopeParts = this.scope.split("/").filter((part) => part.length > 0);
      for (const part of scopeParts) {
        directory = await directory.getDirectoryHandle(
          encodeURIComponent(part),
          { create: true },
        );
      }
      return directory;
    })();
    return this.directoryPromise;
  }

  private async readMetadata(
    fileHandle: OPFSFileHandle,
  ): Promise<OPFSStoreMetadata | undefined> {
    try {
      const text = await (await fileHandle.getFile()).text();
      const parsed = JSON.parse(text);
      if (
        parsed === undefined ||
        parsed === null ||
        typeof parsed !== "object" ||
        typeof parsed.url !== "string"
      ) {
        throw new Error("OPFSStore: invalid metadata format.");
      }
      const metadata: OPFSStoreMetadata = {
        url: parsed.url,
      };
      if (typeof parsed.contentType === "string") {
        metadata.contentType = parsed.contentType;
      }
      return metadata;
    } catch (err) {
      if (this.isNotFoundError(err)) {
        // Treat metadata disappearance between lookup and read as a cache miss
        return undefined;
      }
      throw err;
    }
  }

  private async writeFile(
    handle: OPFSFileHandle,
    data: Blob | BufferSource | string,
  ): Promise<void> {
    const writable = await handle.createWritable();
    await writable.write(data);
    await writable.close();
  }

  private async getFileHandleIfExists(
    directory: OPFSDirectoryHandle,
    filename: string,
    create: boolean,
  ): Promise<OPFSFileHandle | undefined> {
    try {
      return await directory.getFileHandle(filename, { create });
    } catch (err) {
      if (this.isNotFoundError(err)) {
        // NotFound maps to cache miss semantics
        return undefined;
      }
      throw err;
    }
  }

  private async removeEntryIfExists(
    directory: OPFSDirectoryHandle,
    filename: string,
  ): Promise<void> {
    try {
      await directory.removeEntry(filename);
    } catch (err) {
      if (this.isNotFoundError(err)) {
        // Delete is intentionally idempotent for missing entries
        return;
      }
      throw err;
    }
  }

  private async hashUrl(url: string): Promise<string> {
    const textEncoder = new TextEncoder();
    const input = textEncoder.encode(url);
    if (
      typeof crypto === "undefined" ||
      crypto.subtle === undefined ||
      typeof crypto.subtle.digest !== "function"
    ) {
      throw new Error("OPFSStore: crypto.subtle.digest is unavailable.");
    }
    const digest = await crypto.subtle.digest(HASH_ALGORITHM, input);
    return Array.from(new Uint8Array(digest))
      .map((byte) => byte.toString(16).padStart(2, "0"))
      .join("");
  }

  private isNotFoundError(err: unknown): boolean {
    if (err && typeof err === "object" && "name" in err) {
      const name = (err as { name?: unknown }).name;
      return name === "NotFoundError";
    }
    return false;
  }
}
