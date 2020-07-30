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
    else if (dummy == std::string("float64")) return H5T_NATIVE_INT;
    else if (dummy == std::string("int64")) return H5T_NATIVE_LLONG;
    else return H5T_C_S1;
}

///\name HDF class to manage writing information
class H5OutputFile
{

private:

    hid_t file_id;
#ifdef USEPARALLELHDF
    hid_t parallel_access_id;
#endif

protected:

    // Called if a HDF5 call fails (might need to MPI_Abort)
    void io_error(std::string message) {
        std::cerr << message << std::endl;
#ifdef USEMPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        abort();
    }
    unsigned int HDFOUTPUTCHUNKSIZE = 8192;
public:

    // Constructor
    H5OutputFile() {
        file_id = -1;
#ifdef USEPARALLELHDF
        parallel_access_id = -1;
#endif
    }

    // Create a new file
    void create(std::string filename, hid_t flag = H5F_ACC_TRUNC,
        int taskID = -1, bool iparallelopen = true);

    void append(std::string filename, hid_t flag = H5F_ACC_RDWR,
        int taskID = -1, bool iparallelopen = true);

    // Close the file
    void close();

    hid_t create_group(std::string groupname) {
        hid_t group_id = H5Gcreate(file_id, groupname.c_str(),
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        return group_id;
    }
    herr_t close_group(hid_t gid) {
        herr_t status = H5Gclose(gid);
        return status;
    }

    //????
    hid_t create_dataset(std::string dsetname, hid_t file) {
        hid_t dset_id H5Dcreate(file_id, datasetname, -1, -1, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        hid_t group_id = H5Dcreate(file_id, groupname.c_str(),
            H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        return group_id;
    }
    herr_t close_dataset(hid_t did) {
        herr_t status = H5Gclose(gid);
        return status;
    }

    // Destructor closes the file if it's open
    ~H5OutputFile()
    {
        if(file_id >= 0) close();
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

    /// write an attribute
    template <typename T> void write_attribute(std::string parent, std::string name, std::vector<T> data);

    template <typename T> void write_attribute(std::string parent, std::string name, T data);

    void write_attribute(std::string parent, std::string name, std::string data);
};

#endif
