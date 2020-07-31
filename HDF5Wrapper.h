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
    template<typename T> void _do_read(const hid_t &attr, const hid_t &type, T &val)
    {
        if (hdf5_type(T{}) == H5T_C_S1)
        {
          _do_read_string(attr, type, val);
        }
        else {
          H5Aread(attr, type, &val);
        }
    }
    /// wrapper for reading string
    void _do_read_string(const hid_t &attr, const hid_t &type, std::string &val);
    /// wrapper for reading vector
    template<typename T> void _do_read_v(const hid_t &attr, const hid_t &type, std::vector<T> &val)
    {
        hid_t space = H5Aget_space (attr);
        int npoints = H5Sget_simple_extent_npoints(space);
        val.resize(npoints);
        H5Aread(attr, type, val.data());
        H5Sclose(space);
    }

    /// get dataset id
    void _get_dataset(std::vector<hid_t> &ids, const std::string dset_name);
    /// get attribute id from tokenized string
    void _get_dataset(std::vector<hid_t> &ids, const std::vector<std::string> &parts);

public:

    // Constructor
    H5OutputFile();
    // Destructor closes the file if it's open
    ~H5OutputFile();

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
    hid_t open_group(std::string groupname) {
        hid_t group_id = H5Gopen(file_id, groupname.c_str(), H5P_DEFAULT);
        return group_id;
    }
    /// close group
    herr_t close_group(hid_t gid) {
        herr_t status = H5Gclose(gid);
        return status;
    }

    /// close path
    void close_path(std::string path);
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

    template <typename T> hid_t create_dataset(std::string path, T&data,
      std::vector<hsize_t> dims, const std::vector<hsize_t>& chunkDims,
      bool flag_parallel = true, bool flag_hyperslab = true, bool flag_collective = true)
    {
      return create_dataset(path, hdf5_type(T{}), dims, chunkDims,
      flag_parallel, flag_hyperslab, flag_collective);
    }
    hid_t create_dataset(std::string path, hid_t datatype,
      std::vector<hsize_t> dims, const std::vector<hsize_t>& chunkDims,
      bool flag_parallel = true, bool flag_hyperslab = true, bool flag_collective = true);
    /// close data set
    herr_t close_dataset(hid_t dset_id) {
        herr_t status = H5Dclose(dset_id);
        return status;
    }


    //write 1D data sets of string or input data with defined data type
    void write_dataset(std::string name, hsize_t len, std::string data,
      bool flag_parallel = true, bool flag_collective = true);
    void write_dataset(std::string name, hsize_t len, void *data,
       hid_t memtype_id=-1, hid_t filetype_id=-1,
       bool flag_parallel = true, bool flag_first_dim_parallel = true,
       bool flag_hyperslab = true, bool flag_collective = true);

  /// Write a new 1D dataset. Data type of the new dataset is taken to be the type of
  /// the input data if not explicitly specified with the filetype_id parameter.
  /// template function so defined here
  template <typename T> void write_dataset(std::string name, hsize_t len, T *data,
     hid_t memtype_id = -1, hid_t filetype_id=-1,
     bool flag_parallel = true, bool flag_hyperslab = true, bool flag_collective = true)
  {
     int rank = 1;
     hsize_t dims[1] = {len};
     if (memtype_id == -1) memtype_id = hdf5_type(T{});
     write_dataset_nd(name, rank, dims, data, memtype_id, filetype_id, flag_parallel, flag_hyperslab, flag_collective);
  }

  /// Write a multidimensional dataset. Data type of the new dataset is taken to be the type of
  /// the input data if not explicitly specified with the filetype_id parameter.
  template <typename T> void write_dataset_nd(std::string name, int rank, hsize_t *dims, T *data,
      hid_t memtype_id = -1, hid_t filetype_id = -1,
      bool flag_parallel = true, bool flag_first_dim_parallel = true,
      bool flag_hyperslab = true, bool flag_collective = true)
  {
#ifdef USEPARALLELHDF
    MPI_Comm comm = mpi_comm_write;
    MPI_Info info = MPI_INFO_NULL;
#endif
    hid_t dspace_id, dset_id, prop_id, memspace_id, ret;
    std::vector<hsize_t> chunks(rank);

    // Get HDF5 data type of the array in memory
    if (memtype_id == -1) memtype_id = hdf5_type(T{});

    // Determine type of the dataset to create
    if(filetype_id < 0) filetype_id = memtype_id;

#ifdef USEPARALLELHDF
    std::vector<unsigned long long> mpi_hdf_dims(rank*NProcsWrite), mpi_hdf_dims_tot(rank), dims_single(rank), dims_offset(rank);
    if (flag_parallel) {
        //if parallel hdf5 get the full extent of the data
        //this bit of code communicating information can probably be done elsewhere
        //minimize number of mpi communications
        for (auto i=0;i<rank;i++) dims_single[i]=dims[i];
        MPI_Allgather(dims_single.data(), rank, MPI_UNSIGNED_LONG_LONG, mpi_hdf_dims.data(), rank, MPI_UNSIGNED_LONG_LONG, comm);
        MPI_Allreduce(dims_single.data(), mpi_hdf_dims_tot.data(), rank, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
        for (auto i=0;i<rank;i++) {
            dims_offset[i] = 0;
            if (flag_first_dim_parallel && i > 0) continue;
            for (auto j=1;j<=ThisWriteTask;j++) {
                dims_offset[i] += mpi_hdf_dims[i*NProcs+j-1];
            }
        }
        if (flag_first_dim_parallel && rank > 1) {
            for (auto i=1; i<rank;i++) mpi_hdf_dims_tot[i] = dims[i];
        }
    }
#endif

    // Determine if going to compress data in chunks
    // Only chunk non-zero size datasets
    int nonzero_size = 1;
    for(int i=0; i<rank; i++)
    {
#ifdef USEPARALLELHDF
        if (flag_parallel) {
            if(mpi_hdf_dims_tot[i]==0) nonzero_size = 0;
        }
        else {
            if(dims[i]==0) nonzero_size = 0;
        }
#else
        if(dims[i]==0) nonzero_size = 0;
#endif
    }
    // Only chunk datasets where we would have >1 chunk
    int large_dataset = 0;
    for(int i=0; i<rank; i++)
    {
#ifdef USEPARALLELHDF
        if (flag_parallel) {
            if(mpi_hdf_dims_tot[i] > HDFOUTPUTCHUNKSIZE) large_dataset = 1;
        }
        else {
            if(dims[i] > HDFOUTPUTCHUNKSIZE) large_dataset = 1;
        }
#else
        if(dims[i] > HDFOUTPUTCHUNKSIZE) large_dataset = 1;
#endif
    }
    if(nonzero_size && large_dataset)
    {
#ifdef USEPARALLELHDF
        if (flag_parallel) {
            for(auto i=0; i<rank; i++) chunks[i] = std::min((hsize_t) HDFOUTPUTCHUNKSIZE, mpi_hdf_dims_tot[i]);
        }
        else {
            for(auto i=0; i<rank; i++) chunks[i] = std::min((hsize_t) HDFOUTPUTCHUNKSIZE, dims[i]);
        }
#else
        for(auto i=0; i<rank; i++) chunks[i] = std::min((hsize_t) HDFOUTPUTCHUNKSIZE, dims[i]);
#endif
    }

    // Create the dataspace
#ifdef USEPARALLELHDF
    if (flag_parallel) {
        //then all threads create the same simple data space
        //so the meta information is the same
        if (flag_hyperslab) {
            //allocate the space spanning the file
            dspace_id = H5Screate_simple(rank, mpi_hdf_dims_tot.data(), NULL);
            //allocate the memory space
            //allocate the memory space
            memspace_id = H5Screate_simple(rank, dims, NULL);
        }
        else {
            dspace_id = H5Screate_simple(rank, dims, NULL);
            memspace_id = dspace_id;
        }
    }
    else {
        dspace_id = H5Screate_simple(rank, dims, NULL);
        memspace_id = dspace_id;
    }
#else
    dspace_id = H5Screate_simple(rank, dims, NULL);
    memspace_id = dspace_id;
#endif

    // Dataset creation properties
    prop_id = H5P_DEFAULT;
#ifdef USEHDFCOMPRESSION
    // this defines compression
    if(nonzero_size && large_dataset)
    {
        prop_id = H5Pcreate(H5P_DATASET_CREATE);
        H5Pset_layout(prop_id, H5D_CHUNKED);
        H5Pset_chunk(prop_id, rank, chunks.data());
        H5Pset_deflate(prop_id, HDFDEFLATE);
    }
#endif
    // Create the dataset
    dset_id = H5Dcreate(file_id, name.c_str(), filetype_id, dspace_id,
        H5P_DEFAULT, prop_id, H5P_DEFAULT);
    if(dset_id < 0) io_error(std::string("Failed to create dataset: ")+name);
    H5Pclose(prop_id);

    prop_id = H5P_DEFAULT;
#ifdef USEPARALLELHDF
    if (flag_parallel) {
        // set up the collective transfer properties list
        prop_id = H5Pcreate(H5P_DATASET_XFER);
        //if all tasks are participating in the writes
        if (flag_collective) ret = H5Pset_dxpl_mpio(prop_id, H5FD_MPIO_COLLECTIVE);
        else ret = H5Pset_dxpl_mpio(prop_id, H5FD_MPIO_INDEPENDENT);
        if (flag_hyperslab) {
            H5Sselect_hyperslab(dspace_id, H5S_SELECT_SET, dims_offset.data(), NULL, dims, NULL);
            if (dims[0] == 0) {
                H5Sselect_none(dspace_id);
                H5Sselect_none(memspace_id);
            }

        }
        if (mpi_hdf_dims_tot[0] > 0) {
            // Write the data
            ret = H5Dwrite(dset_id, memtype_id, memspace_id, dspace_id, prop_id, data);
            if (ret < 0) io_error(std::string("Failed to write dataset: ")+name);
        }
    }
    else if (dims[0] > 0)
    {
        // Write the data
        ret = H5Dwrite(dset_id, memtype_id, memspace_id, dspace_id, prop_id, data);
        if (ret < 0) io_error(std::string("Failed to write dataset: ")+name);
    }

#else
    // Write the data
    if (dims[0] > 0) {
        ret = H5Dwrite(dset_id, memtype_id, memspace_id, dspace_id, prop_id, data);
        if (ret < 0) io_error(std::string("Failed to write dataset: ")+name);
    }
#endif

    // Clean up (note that dtype_id is NOT a new object so don't need to close it)
    H5Pclose(prop_id);
#ifdef USEPARALLELHDF
    if (flag_hyperslab && flag_parallel) H5Sclose(memspace_id);
#endif
    H5Sclose(dspace_id);
    H5Dclose(dset_id);
  }

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
    template<typename T> const T read_attribute(const std::string &name)
    {
        std::string attr_name;
        T val;
        hid_t type;
        H5O_info_t object_info;
        std::vector <hid_t> ids;
        //traverse the file to get to the attribute, storing the ids of the
        //groups, data spaces, etc that have been opened.
        get_attribute(ids, name);
        //now reverse ids and load attribute
        reverse(ids.begin(),ids.end());
        //determine hdf5 type of the array in memory
        type = hdf5_type(T{});
        // read the data
        _do_read<T>(ids[0], type, val);
        H5Aclose(ids[0]);
        ids.erase(ids.begin());
        //now have hdf5 ids traversed to get to desired attribute so move along to close all
        //based on their object type
        close_hdf_ids(ids);
        return val;
    }
    /// read vector attribute
    template<typename T> const std::vector<T> read_attribute_v(const std::string &name)
    {
        std::string attr_name;
        std::vector<T> val;
        hid_t type;
        H5O_info_t object_info;
        std::vector <hid_t> ids;
        //traverse the file to get to the attribute, storing the ids of the
        //groups, data spaces, etc that have been opened.
        get_attribute(ids, name);
        //now reverse ids and load attribute
        reverse(ids.begin(),ids.end());
        //determine hdf5 type of the array in memory
        type = hdf5_type(T{});
        // read the data
        _do_read_v<T>(ids[0], type, val);
        H5Aclose(ids[0]);
        ids.erase(ids.begin());
        //now have hdf5 ids traversed to get to desired attribute so move along to close all
        //based on their object type
        close_hdf_ids(ids);
        return val;
    }

    /// sees if attribute exits
    bool exists_attribute(const std::string &parent, const std::string &name);

    /// write an attribute, not that since these are template function, define here in the header
    template <typename T> void write_attribute(const std::string &parent, const std::string &name, const std::vector<T> &data)
    {
        // Get HDF5 data type of the value to write
        hid_t dtype_id = hdf5_type(data[0]);
        hsize_t size = data.size();

        // Open the parent object
        hid_t parent_id = H5Oopen(file_id, parent.c_str(), H5P_DEFAULT);
        if(parent_id < 0)io_error(std::string("Unable to open object to write attribute: ")+name);

        // Create dataspace
        hid_t dspace_id = H5Screate(H5S_SIMPLE);
        hid_t dspace_extent  = H5Sset_extent_simple(dspace_id, 1, &size, NULL);

        // Create attribute
        hid_t attr_id = H5Acreate(parent_id, name.c_str(), dtype_id, dspace_id, H5P_DEFAULT, H5P_DEFAULT);
        if(attr_id < 0)io_error(std::string("Unable to create attribute ")+name+std::string(" on object ")+parent);

        // Write the attribute
        if(H5Awrite(attr_id, dtype_id, data.data()) < 0)
        io_error(std::string("Unable to write attribute ")+name+std::string(" on object ")+parent);

        // Clean up
        H5Aclose(attr_id);
        H5Sclose(dspace_id);
        H5Oclose(parent_id);
    }

    template <typename T> void write_attribute(const std::string &parent, const std::string &name, const T &data)
    {
        // Get HDF5 data type of the value to write
        hid_t dtype_id = hdf5_type(data);

        // Open the parent object
        hid_t parent_id = H5Oopen(file_id, parent.c_str(), H5P_DEFAULT);
        if(parent_id < 0)io_error(std::string("Unable to open object to write attribute: ")+name);

        // Create dataspace
        hid_t dspace_id = H5Screate(H5S_SCALAR);

        // Create attribute
        hid_t attr_id = H5Acreate(parent_id, name.c_str(), dtype_id, dspace_id, H5P_DEFAULT, H5P_DEFAULT);
        if(attr_id < 0)io_error(std::string("Unable to create attribute ")+name+std::string(" on object ")+parent);

        // Write the attribute
        if(H5Awrite(attr_id, dtype_id, &data) < 0)
        io_error(std::string("Unable to write attribute ")+name+std::string(" on object ")+parent);

        // Clean up
        H5Aclose(attr_id);
        H5Sclose(dspace_id);
        H5Oclose(parent_id);
    }

    void write_attribute(const std::string parent, const std::string &name, std::string data)
    {
        // Get HDF5 data type of the value to write
        hid_t dtype_id = H5Tcopy(H5T_C_S1);
        if (data.size() == 0) data=" ";
        H5Tset_size(dtype_id, data.size());
        H5Tset_strpad(dtype_id, H5T_STR_NULLTERM);

        // Open the parent object
        hid_t parent_id = H5Oopen(file_id, parent.c_str(), H5P_DEFAULT);
        if(parent_id < 0)io_error(std::string("Unable to open object to write attribute: ")+name);

        // Create dataspace
        hid_t dspace_id = H5Screate(H5S_SCALAR);

        // Create attribute
        hid_t attr_id = H5Acreate(parent_id, name.c_str(), dtype_id, dspace_id, H5P_DEFAULT, H5P_DEFAULT);
        if(attr_id < 0)io_error(std::string("Unable to create attribute ")+name+std::string(" on object ")+parent);

        // Write the attribute
        if(H5Awrite(attr_id, dtype_id, data.c_str()) < 0)
        io_error(std::string("Unable to write attribute ")+name+std::string(" on object ")+parent);

        // Clean up
        H5Aclose(attr_id);
        H5Sclose(dspace_id);
        H5Oclose(parent_id);
    }

};

#endif
