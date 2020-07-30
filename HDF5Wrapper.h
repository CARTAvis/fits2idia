#ifndef __HDF5WRAPPER_H
#define __HDF5WRAPPER_H

#include "common.h"

#ifdef USEMPI
#include <mpi.h>
#endif

// Overloaded function to return HDF5 type given a C type
static inline hid_t hdf5_type(float dummy)              {return H5T_NATIVE_FLOAT;}
static inline hid_t hdf5_type(double dummy)             {return H5T_NATIVE_DOUBLE;}
static inline hid_t hdf5_type(short dummy)              {return H5T_NATIVE_SHORT;}
static inline hid_t hdf5_type(int dummy)                {return H5T_NATIVE_INT;}
static inline hid_t hdf5_type(long dummy)               {return H5T_NATIVE_LONG;}
static inline hid_t hdf5_type(long long dummy)          {return H5T_NATIVE_LLONG;}
static inline hid_t hdf5_type(unsigned short dummy)     {return H5T_NATIVE_USHORT;}
static inline hid_t hdf5_type(unsigned int dummy)       {return H5T_NATIVE_UINT;}
static inline hid_t hdf5_type(unsigned long dummy)      {return H5T_NATIVE_ULONG;}
static inline hid_t hdf5_type(unsigned long long dummy) {return H5T_NATIVE_ULLONG;}
static inline hid_t hdf5_type(std::string dummy)        {return H5T_C_S1;}
static inline hid_t hdf5_type_from_string(std::string dummy)
{
    if (dummy == std::string("float32")) return H5T_NATIVE_FLOAT;
    else if (dummy == std::string("float64")) return H5T_NATIVE_DOUBLE;
    else if (dummy == std::string("int16")) return H5T_NATIVE_SHORT;
    else if (dummy == std::string("int32")) return H5T_NATIVE_INT;
    else if (dummy == std::string("int64")) return H5T_NATIVE_LLONG;
    else if (dummy == std::string("uint16")) return H5T_NATIVE_USHORT;
    else if (dummy == std::string("uint32")) return H5T_NATIVE_UINT;
    else if (dummy == std::string("uint64")) return H5T_NATIVE_ULLONG;
    else return H5T_C_S1;
}

#if H5_VERSION_GE(1,10,1)
#endif

#if H5_VERSION_GE(1,12,0)
#endif

///\name HDF class to manage writing information
class H5OutputFile
{

private:

    hid_t file_id;
#ifdef USEPARALLELHDF
    hid_t parallel_access_id;
#endif

protected:

    /// size of chunks when compressing
    unsigned int HDFOUTPUTCHUNKSIZE = 8192;

    /// Called if a HDF5 call fails (might need to MPI_Abort)
    void io_error(std::string message) {
        std::cerr << message << std::endl;
#ifdef USEMPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        abort();
    }

    /// tokenize a path given an input string
    std::vector<std::string> _tokenize(const std::string &s);
    /// get attribute id
    void _get_attribute(std::vector<hid_t> &ids, const std::string attr_name);
    /// get attribute id from tokenized string
    void _get_attribute(std::vector<hid_t> &ids, const std::vector<std::string> &parts);
    /// wrapper for reading scalar
    template<typename T> void _do_read(const hid_t &attr, const hid_t &type, T &val);
    /// wrapper for reading string
    void _do_read_string(const hid_t &attr, const hid_t &type, std::string &val);
    /// wrapper for reading vector
    template<typename T> void _do_read_v(const hid_t &attr, const hid_t &type, std::vector<T> &val);

    /// get dataset id
    void _get_dataset(std::vector<hid_t> &ids, const std::string dset_name);
    /// get attribute id from tokenized string
    void _get_dataset(std::vector<hid_t> &ids, const std::vector<std::string> &parts);

public:

    // Constructor
    H5OutputFile() {
        file_id = -1;
#ifdef USEPARALLELHDF
        parallel_access_id = -1;
#endif
    }

    // Destructor closes the file if it's open
    ~H5OutputFile()
    {
        if(file_id >= 0) close();
    }

    /// Create a new file
    void create(std::string filename, hid_t flag = H5F_ACC_TRUNC,
        int taskID = -1, bool iparallelopen = true);
    /// Append to a file
    void append(std::string filename, hid_t flag = H5F_ACC_RDWR,
        int taskID = -1, bool iparallelopen = true);

    /// Close the file
    void close();

    /// create a group
    hid_t create_group(std::string groupname) {
        hid_t group_id = H5Gcreate(file_id, groupname.c_str(),
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        return group_id;
    }
    /// close group
    herr_t close_group(hid_t gid) {
        herr_t status = H5Gclose(gid);
        return status;
    }

    /// close hids stored in vector
    void close_hdf_ids(std::vector<hid_t> &ids);

    /// create a dataspace
    hid_t create_dataspace(const std::vector<hsize_t> &dims)
    {
        hid_t dspace_id;
        int rank = dims.size();
        dspace_id = H5Screate_simple(rank, dims.data(), NULL);
        return dspace_id;
    }
    hid_t create_dataspace(int rank, hsize_t *dims)
    {
        hid_t dspace_id;
        dspace_id = H5Screate_simple(rank, dims, NULL);
        return dspace_id;
    }
    hid_t create_dataspace(hsize_t len)
    {
        int rank = 1;
        hsize_t dims[1] = {len};
        hid_t dspace_id = H5Screate_simple(rank, dims, NULL);
        return dspace_id;
    }
    /// close data set
    herr_t close_dataspace(hid_t dspace_id) {
        herr_t status = H5Dclose(dspace_id);
        return status;
    }

    /// create a dataset
    hid_t create_dataset(std::string dsetname, hid_t type_id, hid_t dspace_id)
    {
        hid_t dset_id;
        dset_id = H5Dcreate(file_id, dsetname.c_str(), type_id, dspace_id,
          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        return dset_id;
    }
    template <typename T> hid_t create_dataset(std::string dsetname, T *data, hid_t dspace_id)
    {
        hid_t dset_id;
        hid_t type_id = hdf5_type(T{});
        dset_id = H5Dcreate(file_id, dsetname.c_str(), type_id, dspace_id,
          H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        return dset_id;
    }
    /// close data set
    herr_t close_dataset(hid_t dset_id) {
        herr_t status = H5Dclose(dset_id);
        return status;
    }
    /// Write a new 1D dataset. Data type of the new dataset is taken to be the type of
    /// the input data if not explicitly specified with the filetype_id parameter.
    template <typename T> void write_dataset(std::string name, hsize_t len, T *data,
       hid_t memtype_id = -1, hid_t filetype_id=-1, bool flag_parallel = true, bool flag_hyperslab = true, bool flag_collective = true);
    void write_dataset(std::string name, hsize_t len, std::string data, bool flag_parallel = true, bool flag_collective = true);
    void write_dataset(std::string name, hsize_t len, void *data,
       hid_t memtype_id=-1, hid_t filetype_id=-1, bool flag_parallel = true, bool flag_first_dim_parallel = true, bool flag_hyperslab = true, bool flag_collective = true);

    /// Write a multidimensional dataset. Data type of the new dataset is taken to be the type of
    /// the input data if not explicitly specified with the filetype_id parameter.
    template <typename T> void write_dataset_nd(std::string name, int rank, hsize_t *dims, T *data,
        hid_t memtype_id = -1, hid_t filetype_id = -1,
        bool flag_parallel = true, bool flag_first_dim_parallel = true,
        bool flag_hyperslab = true, bool flag_collective = true);
    void write_dataset_nd(std::string name, int rank, hsize_t *dims, void *data,
        hid_t memtype_id = -1, hid_t filetype_id=-1,
        bool flag_parallel = true, bool flag_first_dim_parallel = true,
        bool flag_hyperslab = true, bool flag_collective = true);

    /// get a dataset with full path given by name
    void get_dataset(std::vector<hid_t> &ids, const std::string &name);
    /// check if dataset exits
    bool exists_dataset(const std::string &parent, const std::string &name);

    /// get an attribute with full path given by name
    void get_attribute(std::vector<hid_t> &ids, const std::string &name);
    /// read scalar attribute
    template<typename T> const T read_attribute(const std::string &name);
    /// read vector attribute
    template<typename T> const std::vector<T> read_attribute_v(const std::string &name);
    /// sees if attribute exits
    bool exists_attribute(const std::string &parent, const std::string &name);
    /// write an attribute
    template <typename T> void write_attribute(std::string parent, std::string name, std::vector<T> data);
    template <typename T> void write_attribute(std::string parent, std::string name, T data);
    void write_attribute(std::string parent, std::string name, std::string data);

};

#endif
