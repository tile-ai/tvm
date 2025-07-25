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
/*
 * \file tvm/runtime/data_type.h
 * \brief Primitive runtime data type.
 */
// Acknowledgement: DataType structure design originates from Halide.
#ifndef TVM_RUNTIME_DATA_TYPE_H_
#define TVM_RUNTIME_DATA_TYPE_H_

#include <tvm/runtime/c_runtime_api.h>
#include <tvm/runtime/logging.h>

#include <cstring>
#include <string>
#include <type_traits>

namespace tvm {
namespace runtime {

/*!
 * \brief Runtime primitive data type.
 *
 *  This class is a thin wrapper of DLDataType.
 *  We also make use of DataType in compiler to store quick hint
 */
class DataType {
 public:
  /*!
   * \brief Type code for the DataType.
   *
   * DLPack consistency:
   * 1) kInt is consistent with kDLInt
   * 2) kUInt is consistent with kDLUInt
   * 3) kFloat is consistent with kDLFloat
   */
  enum TypeCode {
    kInt = kDLInt,
    kUInt = kDLUInt,
    kFloat = kDLFloat,
    kHandle = TVMArgTypeCode::kTVMOpaqueHandle,
    kBFloat = kDLBfloat,
    kFloat8_e4m3fn = 6U,
    kFloat8_e4m3fnuz = 23,
    kFloat8_e5m2 = 7U,
    kFloat4_e2m1fn = 8U,
    kCustomBegin = 129
  };
  /*! \brief default constructor */
  DataType() { data_ = DataType::Void(); }
  /*!
   * \brief Constructor
   * \param dtype The DLDataType
   */
  explicit DataType(DLDataType dtype) : data_(dtype) {}
  /*!
   * \brief Constructor
   * \param code The type code.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes.
   * \param is_scalable Whether the data type is scalable.
   */
  DataType(int code, int bits, int lanes, bool is_scalable = false) {
    data_.code = static_cast<uint8_t>(code);
    data_.bits = static_cast<uint8_t>(bits);
    if (is_scalable) {
      ICHECK(lanes > 1) << "Invalid value for vscale factor" << lanes;
    }
    data_.lanes = is_scalable ? static_cast<uint16_t>(-lanes) : static_cast<uint16_t>(lanes);
    if (code == kBFloat) {
      ICHECK_EQ(bits, 16);
    }
    if (code == kFloat8_e4m3fn || code == kFloat8_e5m2 ||
        code == kFloat8_e4m3fnuz) {
      ICHECK_EQ(bits, 8);
    }
  }
  /*! \return The type code. */
  int code() const { return static_cast<int>(data_.code); }
  /*! \return number of bits in the data. */
  int bits() const { return static_cast<int>(data_.bits); }
  /*! \return number of bytes to store each scalar. */
  int bytes() const { return (bits() + 7) / 8; }
  /*! \return number of lanes in the data. */
  int lanes() const {
    int lanes_as_int = static_cast<int16_t>(data_.lanes);
    if (lanes_as_int < 0) {
      LOG(FATAL) << "Can't fetch the lanes of a scalable vector at a compile time.";
    }
    return lanes_as_int;
  }
  /*! \return the integer multiplier of vscale in a scalable vector. */
  int vscale_factor() const {
    int lanes_as_int = static_cast<int16_t>(data_.lanes);
    if (lanes_as_int >= -1) {
      LOG(FATAL) << "A fixed length vector doesn't have a vscale factor.";
    }
    return -lanes_as_int;
  }
  /*! \return get vscale factor or lanes depending on scalability of the vector. */
  int get_lanes_or_vscale_factor() const {
    return is_scalable_vector() ? vscale_factor() : lanes();
  }
  /*! \return whether type is a scalar type. */
  bool is_scalar() const { return !is_scalable_vector() && lanes() == 1; }
  /*! \return whether type is a scalar type. */
  bool is_bool() const { return code() == DataType::kUInt && bits() == 1; }
  /*! \return whether type is a float type. */
  bool is_float() const { return code() == DataType::kFloat; }
  /*! \return whether type is a float8 type. */
  bool is_float4() const { return code() == DataType::kFloat4_e2m1fn && bits() == 4; }
  bool is_float4_e2m1fn() const { return (code() == DataType::kFloat4_e2m1fn && bits() == 4); }
  bool is_float8() const {
    return (code() == DataType::kFloat || code() == DataType::kFloat8_e4m3fn || code() == DataType::kFloat8_e4m3fnuz ||
            code() == DataType::kFloat8_e5m2) &&
           bits() == 8;
  }
  bool is_float8_e4m3fn() const { return (code() == DataType::kFloat8_e4m3fn && bits() == 8); }
  bool is_float8_e4m3fnuz() const {
    return (code() == DataType::kFloat8_e4m3fnuz && bits() == 8);
  }
  bool is_float8_e5m2() const { return (code() == DataType::kFloat8_e5m2 && bits() == 8); }
  bool is_e4m3_float8() const { return (code() == DataType::kFloat8_e4m3fn && bits() == 8); }
  bool is_e4m3fnuz_float8() const {
    return (code() == DataType::kFloat8_e4m3fnuz && bits() == 8);
  }

  bool is_e5m2_float8() const { return (code() == DataType::kFloat8_e5m2 && bits() == 8); }
  /*! \return whether type is a float16 type. */
  bool is_float16() const { return is_float() && bits() == 16; }
  /*! \return whether type is a bfloat16 type. */
  bool is_bfloat16() const { return code() == DataType::kBFloat && bits() == 16; }
  /*! \return whether type is an int type. */
  bool is_int() const { return code() == DataType::kInt; }
  /*! \return whether type is an uint type. */
  bool is_uint() const { return code() == DataType::kUInt; }
  /*! \return whether type is a handle type. */
  bool is_handle() const { return code() == DataType::kHandle && !is_void(); }
  /*! \return whether type is a vector type. */
  bool is_scalable_or_fixed_length_vector() const {
    int encoded_lanes = static_cast<int16_t>(data_.lanes);
    return (encoded_lanes < -1) || (1 < encoded_lanes);
  }
  /*! \return Whether the type is a fixed length vector. */
  bool is_fixed_length_vector() const { return static_cast<int16_t>(data_.lanes) > 1; }
  /*! \return Whether the type is a scalable vector. */
  bool is_scalable_vector() const { return static_cast<int16_t>(data_.lanes) < -1; }
  /*! \return whether type is a vector type. */
  bool is_vector() const { return lanes() > 1; }
  /*! \return whether type is a bool vector type. */
  bool is_vector_bool() const { return is_scalable_or_fixed_length_vector() && bits() == 1; }
  /*! \return whether type is a Void type. */
  bool is_void() const { return code() == DataType::kHandle && bits() == 0 && lanes() == 0; }
  /*!
   * \brief Create a new data type by change lanes to a specified value.
   * \param lanes The target number of lanes.
   * \return the result type.
   */
  DataType with_lanes(int lanes) const { return DataType(data_.code, data_.bits, lanes); }
  /*!
   * \brief Create a new scalable vector data type by changing the vscale multiplier to a specified
   * value. We'll use the data_.lanes field for this value. \param vscale_factor The vscale
   * multiplier. \return A copy of the old DataType with the number of scalable lanes.
   */
  DataType with_scalable_vscale_factor(int vscale_factor) const {
    return DataType(data_.code, data_.bits, -vscale_factor);
  }
  /*!
   * \brief Create a new data type by change bits to a specified value.
   * \param bits The target number of bits.
   * \return the result type.
   */
  DataType with_bits(int bits) const { return DataType(data_.code, bits, data_.lanes); }
  /*!
   * \brief Get the scalar version of the type.
   * \return the result type.
   */
  DataType element_of() const { return with_lanes(1); }
  /*!
   * \brief Assignment operator.
   */
  DataType& operator=(const DataType& rhs) {
    if (this == &rhs) {
      return *this;
    }
    data_ = rhs.data_;
    return *this;
  }
  /*!
   * \brief Equal comparator.
   * \param other The data type to compare against.
   * \return The comparison result.
   */
  bool operator==(const DataType& other) const {
    return data_.code == other.data_.code && data_.bits == other.data_.bits &&
           data_.lanes == other.data_.lanes;
  }
  /*!
   * \brief NotEqual comparator.
   * \param other The data type to compare against.
   * \return The comparison result.
   */
  bool operator!=(const DataType& other) const { return !operator==(other); }
  /*!
   * \brief Converter to DLDataType
   * \return the result.
   */
  operator DLDataType() const { return data_; }

  /*!
   * \brief Construct an int type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes.
   * \return The constructed data type.
   */
  static DataType Int(int bits, int lanes = 1) { return DataType(kDLInt, bits, lanes); }
  /*!
   * \brief Construct an uint type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes.
   * \param is_scalable Whether the data type is scalable.
   * \return The constructed data type.
   */
  static DataType UInt(int bits, int lanes = 1, bool is_scalable = false) {
    return DataType(kDLUInt, bits, lanes, is_scalable);
  }
  /*!
   * \brief Construct an float type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType Float(int bits, int lanes = 1) { return DataType(kDLFloat, bits, lanes); }
  /*!
   * \brief Construct an bfloat type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType BFloat(int bits, int lanes = 1) { return DataType(kDLBfloat, bits, lanes); }
  /*!
   * \brief Construct NV float8 e4m3 datatype.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType NVFloat8E4M3(int lanes = 1) { return DataType(kFloat8_e4m3fn, 8, lanes); }
  /*!
   * \brief Construct NV float8 e4m3fnuz datatype.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType NVFloat8E4M3UZN(int lanes = 1) { return DataType(kFloat8_e4m3fnuz, 8, lanes); }
  
  /*!
   * \brief Construct NV float8 e5m2 datatype.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType NVFloat8E5M2(int lanes = 1) { return DataType(kFloat8_e5m2, 8, lanes); }
  /*!
   * \brief Construct a bool type.
   * \param lanes The number of lanes.
   * \param is_scalable Whether the data type is scalable.
   * \return The constructed data type.
   */
  static DataType Bool(int lanes = 1, bool is_scalable = false) {
    return DataType::UInt(1, lanes, is_scalable);
  }
  /*!
   * \brief Construct a handle type.
   * \param bits The number of bits in the type.
   * \param lanes The number of lanes
   * \return The constructed data type.
   */
  static DataType Handle(int bits = 64, int lanes = 1) { return DataType(kHandle, bits, lanes); }
  /*!
   * \brief Construct a Void type.
   * \return The constructed data type.
   */
  static DataType Void() { return DataType(kHandle, 0, 0); }
  /*!
   * \brief Get the corresponding type of TVMShapeIndex.
   * \return The type of TVM shape index.
   */
  static DataType ShapeIndex() {
    if (std::is_signed<tvm_index_t>::value) {
      return DataType::Int(sizeof(tvm_index_t) * 8);
    } else {
      return DataType::UInt(sizeof(tvm_index_t) * 8);
    }
  }

 private:
  DLDataType data_;
};

/*!
 * \brief Get the number of bytes needed in a vector.
 * \param dtype The data type.
 * \return Number of bytes needed.
 */
inline int GetVectorBytes(DataType dtype) {
  int data_bits = dtype.bits() * dtype.lanes();
  // allow bool to exist
  if (dtype == DataType::Bool() || dtype == DataType::Int(4) || dtype == DataType::UInt(4) ||
      dtype == DataType::Int(1)) {
    return 1;
  }
  ICHECK_EQ(data_bits % 8, 0U) << "Need to load/store by multiple of bytes";
  return data_bits / 8;
}

/*!
 * \brief Check whether type matches the given spec.
 * \param t The type
 * \param code The type code.
 * \param bits The number of bits to be matched.
 * \param lanes The number of lanes in the type.
 */
inline bool TypeMatch(DLDataType t, int code, int bits, int lanes = 1) {
  return t.code == code && t.bits == bits && t.lanes == lanes;
}
/*!
 * \brief Check whether two types are equal .
 * \param lhs The left operand.
 * \param rhs The right operand.
 */
inline bool TypeEqual(DLDataType lhs, DLDataType rhs) {
  return lhs.code == rhs.code && lhs.bits == rhs.bits && lhs.lanes == rhs.lanes;
}

/*!
 * \brief Runtime utility for getting custom type name from code
 * \param type_code Custom type code
 * \return Custom type name
 */
TVM_DLL std::string GetCustomTypeName(uint8_t type_code);

/*!
 * \brief Runtime utility for checking whether custom type is registered
 * \param type_code Custom type code
 * \return Bool representing whether type is registered
 */
TVM_DLL bool GetCustomTypeRegistered(uint8_t type_code);

/*!
 * \brief Runtime utility for parsing string of the form "custom[<typename>]"
 * \param s String to parse
 * \param scan pointer to parsing pointer, which is scanning across s
 * \return type code of custom type parsed
 */
TVM_DLL uint8_t ParseCustomDatatype(const std::string& s, const char** scan);

/*!
 * \brief Convert type code to its name
 * \param type_code The type code .
 * \return The name of type code.
 */
inline const char* DLDataTypeCode2Str(DLDataTypeCode type_code);

/*!
 * \brief convert a string to TVM type.
 * \param s The string to be converted.
 * \return The corresponding tvm type.
 */
inline DLDataType String2DLDataType(std::string s);

/*!
 * \brief convert a TVM type to string.
 * \param t The type to be converted.
 * \return The corresponding tvm type in string.
 */
inline std::string DLDataType2String(DLDataType t);

// implementation details
inline const char* DLDataTypeCode2Str(DLDataTypeCode type_code) {
  switch (static_cast<int>(type_code)) {
    case kDLInt:
      return "int";
    case kDLUInt:
      return "uint";
    case kDLFloat:
      return "float";
    case DataType::kHandle:
      return "handle";
    case kDLBfloat:
      return "bfloat";
    case DataType::kFloat8_e4m3fn:
      return "e4m3_float";
    case DataType::kFloat8_e5m2:
      return "e5m2_float";
    case DataType::kFloat8_e4m3fnuz:
      return "e4m3fnuz_float";
    default:
      LOG(FATAL) << "unknown type_code=" << static_cast<int>(type_code);
  }
  throw;
}

inline std::ostream& operator<<(std::ostream& os, DLDataType t) {  // NOLINT(*)
  if (t.bits == 1 && t.lanes == 1 && t.code == kDLUInt) {
    os << "bool";
    return os;
  }
  if (DataType(t).is_void()) {
    return os << "void";
  }
  if (t.code < DataType::kCustomBegin) {
    os << DLDataTypeCode2Str(static_cast<DLDataTypeCode>(t.code));
  } else {
    os << "custom[" << GetCustomTypeName(t.code) << "]";
  }
  if (t.code == kTVMOpaqueHandle) return os;
  int16_t lanes = static_cast<int16_t>(t.lanes);
  os << static_cast<int>(t.bits);
  if (lanes > 1) {
    os << 'x' << lanes;
  } else if (lanes < -1) {
    os << "xvscalex" << -lanes;
  }
  return os;
}

inline std::ostream& operator<<(std::ostream& os, const DataType& dtype) {  // NOLINT(*)
  return os << dtype.operator DLDataType();
}

inline std::string DLDataType2String(DLDataType t) {
  if (t.bits == 0) return "";
  std::ostringstream os;
  os << t;
  return os.str();
}

inline DLDataType String2DLDataType(std::string s) {
  DLDataType t;
  // handle void type
  if (s.length() == 0 || s == "void") {
    t = DataType::Void();
    return t;
  }
  t.bits = 32;
  t.lanes = 1;
  const char* scan;
  
  if (s.substr(0, 3) == "int") {
    t.code = kDLInt;
    scan = s.c_str() + 3;
  } else if (s.substr(0, 4) == "uint") {
    t.code = kDLUInt;
    scan = s.c_str() + 4;
  } else if (s.find("e4m3fnuz") != std::string::npos) {
    t.code = DataType::kFloat8_e4m3fnuz;
    t.bits = 8;
    scan = s.c_str() + 15;
  } else if (s.substr(0, 5) == "float") {
    t.code = kDLFloat;
    scan = s.c_str() + 5;
  } else if (s.substr(0, 6) == "handle") {
    t.code = kTVMOpaqueHandle;
    t.bits = 64;  // handle uses 64 bit by default.
    scan = s.c_str() + 6;
  } else if (s == "bool") {
    t.code = kDLUInt;
    t.bits = 1;
    t.lanes = 1;
    return t;
  } else if (s.substr(0, 6) == "bfloat") {
    t.code = DataType::kBFloat;
    t.bits = 16;
    scan = s.c_str() + 6;
  } else if (s.substr(0, 10) == "e4m3_float") {
    t.code = DataType::kFloat8_e4m3fn;
    t.bits = 8;
    scan = s.c_str() + 10;
  } else if (s.substr(0, 14) == "e4m3fnuz_float") {
    t.code = DataType::kFloat8_e4m3fnuz;
    t.bits = 8;
    scan = s.c_str() + 14;
  } else if (s.substr(0, 10) == "e5m2_float") {
    t.code = DataType::kFloat8_e5m2;
    t.bits = 8;
    scan = s.c_str() + 10;
  } else if (s.substr(0, 6) == "custom") {
    t.code = ParseCustomDatatype(s, &scan);
  } else {
    scan = s.c_str();
    LOG(FATAL) << "unknown type " << s;
  }
  char* xdelim;  // emulate sscanf("%ux%u", bits, lanes)
  uint8_t bits = static_cast<uint8_t>(strtoul(scan, &xdelim, 10));
  if (bits != 0) t.bits = bits;
  int scalable_multiplier = 1;
  if (strncmp(xdelim, "xvscale", 7) == 0) {
    scalable_multiplier = -1;
    xdelim += 7;
  }
  char* endpt = xdelim;
  if (*xdelim == 'x') {
    t.lanes = static_cast<uint16_t>(scalable_multiplier * strtoul(xdelim + 1, &endpt, 10));
  }
  if (endpt != s.c_str() + s.length()) {
    std::string unprocessed_part(endpt); 
    std::cout << "Unknown type: '" << s << "'. Unprocessed part starts at: '" << unprocessed_part << "'";
    ICHECK(endpt == s.c_str() + s.length())
       << "Unknown type: '" << s << "'. Unprocessed part: '" << std::string(endpt) << "'";
  }
  return t;
}

}  // namespace runtime

using DataType = runtime::DataType;

}  // namespace tvm

namespace std {
template <>
struct hash<tvm::DataType> {
  inline int cantor_pairing_function(int a, int b) const { return (a + b) * (a + b + 1) / 2 + b; }
  std::size_t operator()(tvm::DataType const& dtype) const {
    int a = dtype.code();
    int b = dtype.bits();
    int c = dtype.lanes();
    int d = cantor_pairing_function(a, b);
    return cantor_pairing_function(c, d);
  }
};
}  // namespace std

#endif  //  TVM_RUNTIME_DATA_TYPE_H_
