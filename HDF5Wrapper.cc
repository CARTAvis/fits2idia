#include "HDF5Wrapper.h"

// Constructor
H5OutputFile::H5OutputFile()
{
    file_id = -1;
#ifdef USEPARALLELHDF
    parallel_access_id = -1;
#endif
}

// Destructor closes the file if it's open
H5OutputFile::~H5OutputFile()
{
    if(file_id >= 0) close();
}

void H5OutputFile::create(std::string filename, hid_t flag,
    int taskID, bool iparallelopen)
{
    if(file_id >= 0) io_error("Attempted to create file when already open!");
#ifdef USEPARALLELHDF
    MPI_Comm comm = mpi_comm_write;
    MPI_Info info = MPI_INFO_NULL;
    if (iparallelopen && taskID ==-1) {
        parallel_access_id = H5Pcreate (H5P_FILE_ACCESS);
        if (parallel_access_id < 0) io_error("Parallel access creation failed");
        herr_t ret = H5Pset_fapl_mpio(parallel_access_id, comm, info);
        if (ret < 0) io_error("Parallel access failed");
        // create the file collectively
        file_id = H5Fcreate(filename.c_str(), flag, H5P_DEFAULT, parallel_access_id);
        if (file_id < 0) io_error(std::string("Failed to create output file: ")+filename);
        ret = H5Pclose(parallel_access_id);
        if (ret < 0) io_error("Parallel release failed");
        parallel_access_id = -1;
    }
    else {
        if (taskID <0 || taskID > NProcsWrite) io_error(std::string("MPI Task ID asked to create file out of range. Task ID is ")+to_std::string(taskID));
        if (ThisWriteTask == taskID) {
            file_id = H5Fcreate(filename.c_str(), flag, H5P_DEFAULT, H5P_DEFAULT);
            if (file_id < 0) io_error(std::string("Failed to create output file: ")+filename);
            parallel_access_id = -1;
        }
        else {
            parallel_access_id = -2;
        }
        MPI_Barrier(comm);
    }
#else
    file_id = H5Fcreate(filename.c_str(), flag, H5P_DEFAULT, H5P_DEFAULT);
    if(file_id < 0)io_error(std::string("Failed to create output file: ")+filename);
#endif

}

void H5OutputFile::append(std::string filename, hid_t flag,
    int taskID, bool iparallelopen)
{
    if(file_id >= 0)io_error("Attempted to open and append to file when already open!");
#ifdef USEPARALLELHDF
    MPI_Comm comm = mpi_comm_write;
    MPI_Info info = MPI_INFO_NULL;
    if (iparallelopen && taskID ==-1) {
        parallel_access_id = H5Pcreate (H5P_FILE_ACCESS);
        if (parallel_access_id < 0) io_error("Parallel access creation failed");
        herr_t ret = H5Pset_fapl_mpio(parallel_access_id, comm, info);
        if (ret < 0) io_error("Parallel access failed");
        // create the file collectively
        file_id = H5Fopen(filename.c_str(), flag, parallel_access_id);
        if (file_id < 0) io_error(std::string("Failed to create output file: ")+filename);
        ret = H5Pclose(parallel_access_id);
        if (ret < 0) io_error("Parallel release failed");
        parallel_access_id = -1;
    }
    else {
        if (taskID <0 || taskID > NProcsWrite) io_error(std::string("MPI Task ID asked to create file out of range. Task ID is ")+to_std::string(taskID));
        if (ThisWriteTask == taskID) {
            file_id = H5Fopen(filename.c_str(),flag, H5P_DEFAULT);
            if (file_id < 0) io_error(std::string("Failed to create output file: ")+filename);
            parallel_access_id = -1;
        }
        else {
            parallel_access_id = -2;
        }
        MPI_Barrier(comm);
    }
#else
    file_id = H5Fopen(filename.c_str(), flag, H5P_DEFAULT);
    if (file_id < 0) io_error(std::string("Failed to create output file: ")+filename);
#endif
}

// Close the file
void H5OutputFile::close()
{
#ifdef USEPARALLELHDF
    if(file_id < 0 && parallel_access_id == -1) io_error("Attempted to close file which is not open!");
    if (parallel_access_id == -1) H5Fclose(file_id);
#else
    if(file_id < 0) io_error("Attempted to close file which is not open!");
    H5Fclose(file_id);
#endif
    file_id = -1;
#ifdef USEPARALLELHDF
    parallel_access_id = -1;
#endif
}

#ifdef USEPARALLELHDF
void H5OutputFile::_set_mpi_dim_and_offset(MPI_Comm &comm,
    hsize_t rank, std::vector<hsize_t> &dims,
    std::vector<unsigned long long> &dims_single,
    std::vector<unsigned long long> &dims_offset
    std::vector<unsigned long long> &mpi_hdf_dims,
    std::vector<unsigned long long> &mpi_hdf_dims_tot,
    bool flag_parallel, bool flag_first_dim_parallel
    )
{
    if (!flag_parallel) return;
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
void H5OutputFile::_set_mpi_hyperslab(hid_t &dspace_id, hid_t &memspace_id,
    hsize_t rank, std::vector<hsize_t> &dims,
    std::vector<unsigned long long> &mpi_hdf_dims_tot,
    bool flag_parallel, bool flag_hyperslab)
{
    if (flag_parallel) {
        //then all threads create the same simple data space
        //so the meta information is the same
        if (flag_hyperslab) {
            //allocate the space spanning the file
            dspace_id = H5Screate_simple(rank, mpi_hdf_dims_tot.data(), NULL);
            //allocate the memory space
            //allocate the memory space
            memspace_id = H5Screate_simple(rank, dims.data(), NULL);
        }
        else {
            dspace_id = H5Screate_simple(rank, dims.data(), NULL);
            memspace_id = dspace_id;
        }
    }
    else {
        dspace_id = H5Screate_simple(rank, dims.data(), NULL);
        memspace_id = dspace_id;
    }
}

void H5OutputFile::_set_mpi_dataset_properties(hid_t &prop_id, bool &iwrite,
    hid_t &dspace_id, hid_t &memspace_id,
    hsize_t rank, std::vector<hsize_t> dims, std::vector<hsize_t> dims_offset,
    bool flag_parallel, bool flag_collective, bool flag_hyperslab)
{
    if (!flag_parallel) return;
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
    iwrite = (mpi_hdf_dims_tot[0] > 0);
}

#endif

std::vector<std::string> H5OutputFile::_tokenize(const std::string &s)
{
    std::string delims("/");
    std::string::size_type lastPos = s.find_first_not_of(delims, 0);
    std::string::size_type pos     = s.find_first_of(delims, lastPos);

    std::vector<std::string> tokens;
    while (std::string::npos != pos || std::string::npos != lastPos) {
        tokens.push_back(s.substr(lastPos, pos - lastPos));
        lastPos = s.find_first_not_of(delims, pos);
        pos = s.find_first_of(delims, lastPos);
    }
    return tokens;
}

/// get an attribute going to list of hids
void H5OutputFile::_get_attribute(std::vector<hid_t> &ids, const std::string attr_name)
{
    //can use H5Aexists as it is the C interface but how to access it?
    auto exists = H5Aexists(ids.back(), attr_name.c_str());
    if (exists == 0) {
        throw std::invalid_argument(std::string("attribute not found ") + attr_name);
    }
    else if (exists < 0) {
        throw std::runtime_error("Error on H5Aexists");
    }
    auto attr = H5Aopen(ids.back(), attr_name.c_str(), H5P_DEFAULT);
    ids.push_back(attr);
}
/// get attributes parsing list of ids and vector of strings
void H5OutputFile::_get_attribute(std::vector<hid_t> &ids, const std::vector<std::string> &parts)
{
    // This is the attribute name, so open it and store the id
    if (parts.size() == 1) {
        _get_attribute(ids, parts[0]);
    }
    else
    {
        //otherwise enter group and recursively call funciotn
        H5O_info_t object_info;
        hid_t newid;
        H5Oget_info_by_name(ids.back(), parts[0].c_str(), &object_info, H5P_DEFAULT);
        if (object_info.type == H5O_TYPE_GROUP) {
            newid = H5Gopen(ids.back(),parts[0].c_str(),H5P_DEFAULT);
        }
        else if (object_info.type == H5O_TYPE_DATASET) {
            newid = H5Dopen(ids.back(),parts[0].c_str(),H5P_DEFAULT);
        }
        ids.push_back(newid);
        //get the substring
        std::vector<std::string> subparts(parts.begin() + 1, parts.end());
        //call function again
        _get_attribute(ids, subparts);
    }
}

/// get attribute in file, storing relevant ids in vector
void H5OutputFile::get_attribute(std::vector<hid_t> &ids, const std::string &name)
{
    auto parts = _tokenize(name);
    _get_attribute(ids, parts);
}

/// get a dataset going to list of hids
void H5OutputFile::_get_dataset(std::vector<hid_t> &ids, const std::string dset_name)
{
    auto dset_id = H5Dopen(ids.back(), dset_name.c_str(), H5P_DEFAULT);
    if (dset_id < 0) {
        throw std::invalid_argument(std::string("dataset not found ") + dset_name);
    }
    ids.push_back(dset_id);
}
/// get dataset parsing list of ids and vector of strings
void H5OutputFile::_get_dataset(std::vector<hid_t> &ids, const std::vector<std::string> &parts)
{
    // This is the attribute name, so open it and store the id
    if (parts.size() == 1) {
        _get_dataset(ids, parts[0]);
    }
    else
    {
        //otherwise enter group and recursively call funciotn
        H5O_info_t object_info;
        hid_t newid;
        H5Oget_info_by_name(ids.back(), parts[0].c_str(), &object_info, H5P_DEFAULT);
        if (object_info.type == H5O_TYPE_GROUP) {
            newid = H5Gopen(ids.back(),parts[0].c_str(),H5P_DEFAULT);
        }
        else if (object_info.type == H5O_TYPE_DATASET) {
            throw std::runtime_error("Incorrect path to data set, encountered another data set in path");
        }
        ids.push_back(newid);
        //get the substring
        std::vector<std::string> subparts(parts.begin() + 1, parts.end());
        //call function again
        _get_dataset(ids, subparts);
    }
}

/// get dataset in file, storing relevant ids in vector
void H5OutputFile::get_dataset(std::vector<hid_t> &ids, const std::string &name)
{
    auto parts = _tokenize(name);
    _get_dataset(ids, parts);
}

/// get a hdf5 id  going to list of hids
void H5OutputFile::_get_hdf5_id(std::vector<hid_t> &ids, const std::string name)
{
    auto id = H5Oopen(ids.back(), name.c_str(), H5P_DEFAULT);
    if (id < 0) {
        throw std::invalid_argument(std::string("hdf5 object not found ") + name);
    }
    ids.push_back(id);
}
/// get hdf5 ids parsing list of ids and vector of strings
void H5OutputFile::_get_hdf5_id(std::vector<hid_t> &ids, const std::vector<std::string> &parts)
{
    // This is the attribute name, so open it and store the id
    if (parts.size() == 1) {
        _get_hdf5_id(ids, parts[0]);
    }
    else
    {
        //otherwise enter group and recursively call funciotn
        H5O_info_t object_info;
        hid_t newid;
        H5Oget_info_by_name(ids.back(), parts[0].c_str(), &object_info, H5P_DEFAULT);
        if (object_info.type == H5O_TYPE_GROUP) {
            newid = H5Gopen(ids.back(),parts[0].c_str(),H5P_DEFAULT);
        }
        else {
            throw std::runtime_error("Incorrect path, encountered something other than group");
        }
        ids.push_back(newid);
        //get the substring
        std::vector<std::string> subparts(parts.begin() + 1, parts.end());
        //call function again
        _get_dataset(ids, subparts);
    }
}

/// get hdf5 id in file, storing relevant ids in vector
void H5OutputFile::get_hdf5_id(std::vector<hid_t> &ids, const std::string &name)
{
    auto parts = _tokenize(name);
    _get_hdf5_id(ids, parts);
}
hid_t H5OutputFile::get_hdf5_id(std::string path, std::string name, bool closeids)
{
    return get_hdf5_id(path+name, closeids);
}
hid_t H5OutputFile::get_hdf5_id(std::string fullname, bool closeids)
{
    hid_t id;
    std::vector<hid_t> ids;
    try {
        get_hdf5_id(ids, fullname);
        id = ids.back();
    }
    catch (const std::invalid_argument& ia)
    {
        id = -1;
    }
    ///\todo question of whether I should close the objects that have been opened
    if (closeids) close_hdf_ids(ids);
    return id;
}

/// close open hids stored in vector
void H5OutputFile::close_hdf_ids(std::vector<hid_t> &ids)
{
    H5O_info_t object_info;
    for (auto &id:ids)
    {
        H5Oget_info(id, &object_info);
        if (object_info.type == H5O_TYPE_GROUP) {
            H5Gclose(id);
        }
        else if (object_info.type == H5O_TYPE_GROUP) {
            H5Dclose(id);
        }
    }
}

/// close open hids stored in vector
void H5OutputFile::close_path(std::string path)
{
    auto parts = _tokenize(path);
    std::vector<hid_t> ids;
    _get_hdf5_id(ids, parts);
    close_hdf_ids(ids);
}

/// read a string attribute
void H5OutputFile::_do_read_string(const hid_t &attr, const hid_t &type, std::string &val)
{
    std::vector<char> buf;
    hid_t type_in_file = H5Aget_type(attr);
    hid_t type_in_memory = H5Tcopy(type); // copy memory type because we'll need to modify it
    size_t length = H5Tget_size(type_in_file); // get length of the string in the file
    buf.resize(length+1); // resize buffer in memory, allowing for null terminator
    H5Tset_size(type_in_memory, length+1); // tell HDF5 the length of the buffer in memory
    H5Tset_strpad(type_in_memory, H5T_STR_NULLTERM); // specify that we want a null terminated string
    H5Aread(attr, type_in_memory, buf.data());
    H5Tclose(type_in_memory);
    H5Tclose(type_in_file);
    val=std::string(buf.data());
}

bool H5OutputFile::exists_attribute(const std::string &parent, const std::string &name) {
    std::string attr_name = parent+std::string("/")+name;
    std::vector <hid_t> ids;
    bool exists = true;
    //groups, data spaces, etc that have been opened.
    try {
      get_attribute(ids, attr_name);
    }
    catch (const std::invalid_argument& ia)
    {
      exists = false;
    }
    //now reverse ids and load attribute
    reverse(ids.begin(),ids.end());
    //check if present
    if (exists) {
      H5Aclose(ids[0]);
      ids.erase(ids.begin());
    }
    close_hdf_ids(ids);
    return exists;
}

bool H5OutputFile::exists_dataset(const std::string &parent, const std::string &name) {
    std::string dset_name = parent+std::string("/")+name;
    std::vector <hid_t> ids;
    bool exists = true;
    //groups, data spaces, etc that have been opened.
    try {
      get_dataset(ids, dset_name);
    }
    catch (const std::invalid_argument& ia)
    {
      exists = false;
    }
    //now reverse ids and load attribute
    reverse(ids.begin(),ids.end());
    //check if present
    if (exists) {
      H5Dclose(ids[0]);
      ids.erase(ids.begin());
    }
    close_hdf_ids(ids);
    return exists;
}

/// create a dataset
hid_t H5OutputFile::create_dataset(std::string fullname, hid_t type_id,
  std::vector<hsize_t> dims, std::vector<hsize_t> chunkDims,
  bool flag_parallel, bool flag_hyperslab, bool flag_collective)
{
#ifdef USEPARALLELHDF
    MPI_Comm comm = mpi_comm_write;
    MPI_Info info = MPI_INFO_NULL;
#endif
    auto splitPath = _tokenize(fullname);
    auto dsetname = splitPath.back();
    splitPath.pop_back();
    hid_t curr_id = file_id;
    hid_t memspace_id, dspace_id, dset_id, prop_id = H5P_DEFAULT;
    auto rank = dims.size();
    std::vector<hsize_t> chunks = chunkDims;

    /// not certain what to do with the open groups here.
    for (auto& groupname : splitPath)
    {
      if (H5Lexists(curr_id, groupname.c_str(), H5P_DEFAULT) == 0)
      {
        curr_id = create_group(groupname);
      }
      else
      {
        curr_id = open_group(groupname);
      }
    }

#ifdef USEPARALLELHDF
    std::vector<unsigned long long> mpi_hdf_dims(rank*NProcsWrite), mpi_hdf_dims_tot(rank), dims_single(rank), dims_offset(rank);
    if (flag_parallel) _set_mpi_dim_and_offset(comm, rank, dims, dims_single, dims_offset, mpi_hdf_dims, mpi_hdf_dims_tot, flag_first_dim_parallel);
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
    if(nonzero_size && !chunks.empty())
    {
#ifdef USEPARALLELHDF
        if (flag_parallel) {
            for(auto i=0; i<rank; i++) chunks[i] = std::min(chunks[i], mpi_hdf_dims_tot[i]);
        }
        else {
            for(auto i=0; i<rank; i++) chunks[i] = std::min(chunks[i], dims[i]);
        }
#else
        for(auto i=0; i<rank; i++) chunks[i] = std::min(chunks[i], dims[i]);
#endif
    }

    // Create the dataspace
#ifdef USEPARALLELHDF
    _set_mpi_hyperslab(dspace_id, memspace_id, rank, dims, mpi_hdf_dims_tot, flag_parallel, flag_hyperslab);
#else
    dspace_id = H5Screate_simple(rank, dims.data(), NULL);
    memspace_id = dspace_id;
#endif

#ifdef USEHDFCOMPRESSION
    if (!chunk.empty()) {
      prop_id = H5Pcreate(H5P_DATASET_CREATE);
      H5Pset_layout(prop_id, H5D_CHUNKED);
      H5Pset_chunk(prop_id, rank, chunks.data());
      H5Pset_deflate(prop_id, HDFDEFLATE);
    }
#endif

    dset_id = H5Dcreate(file_id, dsetname.c_str(), type_id, dspace_id,
      H5P_DEFAULT, prop_id, H5P_DEFAULT);
    return dset_id;
}

/// create a link
herr_t H5OutputFile::create_link(std::string orgname, std::string linkname, bool ihard) {
    std::vector<hid_t> orgids;
    hid_t orgid = get_hdf5_id(orgname);
    hid_t linkid;

    if (H5Lexists(linkid, linkname.c_str(), H5P_DEFAULT))  {
        throw std::runtime_error("Link already exists, cannot create new link");
    }
    if (ihard) {
        return H5Lcreate_hard(orgid, orgname.c_str(), linkid, linkname.c_str(), H5P_DEFAULT, H5P_DEFAULT);
    }
    else {
        return H5Lcreate_soft(orgname.c_str(), linkid, linkname.c_str(), H5P_DEFAULT, H5P_DEFAULT);
    }
}


void H5OutputFile::write_dataset(std::string name, hsize_t len, std::string data,
    bool flag_parallel, bool flag_collective)
{
#ifdef USEPARALLELHDF
    MPI_Comm comm = mpi_comm_write;
    MPI_Info info = MPI_INFO_NULL;
#endif
    int rank = 1;
    hsize_t dims[1] = {len};

    hid_t memtype_id, filetype_id, dspace_id, dset_id, xfer_plist;
    herr_t status, ret;
    memtype_id = H5Tcopy (H5T_C_S1);
    status = H5Tset_size (memtype_id, data.size());
    filetype_id = H5Tcopy (H5T_C_S1);
    status = H5Tset_size (filetype_id, data.size());

    // Create the dataspace
    dspace_id = H5Screate_simple(rank, dims, NULL);

    // Create the dataset
    dset_id = H5Dcreate(file_id, name.c_str(), filetype_id, dspace_id,
        H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
#ifdef USEPARALLELHDF
    if (flag_parallel) {
        // set up the collective transfer properties list
        xfer_plist = H5Pcreate(H5P_DATASET_XFER);
        if (xfer_plist < 0) io_error(std::string("Failed to set up parallel transfer: ")+name);
        if (flag_collective) ret = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_COLLECTIVE);
        else ret = H5Pset_dxpl_mpio(xfer_plist, H5FD_MPIO_INDEPENDENT);
        if (ret < 0) io_error(std::string("Failed to set up parallel transfer: ")+name);
        // the result of above should be that all processors write to the same
        // point of the hdf file.
    }
#endif
    // Write the data
    if(H5Dwrite(dset_id, memtype_id, dspace_id, H5S_ALL, H5P_DEFAULT, data.c_str()) < 0)
    io_error(std::string("Failed to write dataset: ")+name);

    // Clean up (note that dtype_id is NOT a new object so don't need to close it)
#ifdef USEPARALLELHDF
    if (flag_parallel) H5Pclose(xfer_plist);
#endif
    H5Sclose(dspace_id);
    H5Dclose(dset_id);
}

void H5OutputFile::write_dataset(std::string name, hsize_t len, void *data,
    hid_t memtype_id, hid_t filetype_id,
    bool flag_parallel, bool flag_first_dim_parallel, bool flag_hyperslab, bool flag_collective)
{
    int rank = 1;
    hsize_t dims[1] = {len};
    if (memtype_id == -1) {
        throw std::runtime_error("Write data set called with void pointer but no type info passed.");
    }
    write_dataset_nd(name, rank, dims, data, memtype_id, filetype_id,
        flag_parallel, flag_first_dim_parallel, flag_hyperslab, flag_collective);
}

void H5OutputFile::write_dataset_nd(std::string name, int rank, hsize_t *dims, void *data,
    hid_t memtype_id, hid_t filetype_id,
    bool flag_parallel, bool flag_first_dim_parallel,
    bool flag_hyperslab, bool flag_collective)
{
#ifdef USEPARALLELHDF
    MPI_Comm comm = mpi_comm_write;
    MPI_Info info = MPI_INFO_NULL;
#endif
    hid_t dspace_id, dset_id, prop_id, memspace_id, ret;
    std::vector<hsize_t> chunks(rank);
    // Get HDF5 data type of the array in memory
    if (memtype_id == -1) {
        throw std::runtime_error("Write data set called with void pointer but no type info passed.");
    }
    // Determine type of the dataset to create
    if(filetype_id < 0) filetype_id = memtype_id;

#ifdef USEPARALLELHDF
    std::vector<unsigned long long> mpi_hdf_dims(rank*NProcsWrite), mpi_hdf_dims_tot(rank), dims_single(rank), dims_offset(rank);
    _set_mpi_dim_and_offset(comm, rank, dims, dims_single, dims_offset, mpi_hdf_dims, mpi_hdf_dims_tot, flag_parallel, flag_first_dim_parallel);
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
    unsigned int large_dataset = 0;
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
    _set_mpi_hyperslab(dspace_id, memspace_id, rank, dims, mpi_hdf_dims_tot, flag_parallel, flag_hyperslab);
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
    bool iwrite = (dims[0] > 0);
#ifdef USEPARALLELHDF
    _set_mpi_dataset_properties(prop_id, iwrite,
        dspace_id, memspace_id,
        rank, dims, dims_offset,
        flag_parallel, flag_collective, flag_hyperslab);
#endif
    if (iwrite) {
        ret = H5Dwrite(dset_id, memtype_id, memspace_id, dspace_id, prop_id, data);
        if (ret < 0) io_error(std::string("Failed to write dataset: ")+name);
    }

    // Clean up (note that dtype_id is NOT a new object so don't need to close it)
    H5Pclose(prop_id);
#ifdef USEPARALLELHDF
    if (flag_hyperslab && flag_parallel) H5Sclose(memspace_id);
#endif
    H5Sclose(dspace_id);
    H5Dclose(dset_id);
}

/// Write data set with hyperslab set by count and start
void H5OutputFile::write_dataset(std::string name, hsize_t len, void *data,
    const std::vector<hsize_t>& count, const std::vector<hsize_t>& start,
    hid_t memtype_id, hid_t filetype_id,
    bool flag_parallel, bool flag_first_dim_parallel, bool flag_hyperslab, bool flag_collective)
{
    int rank = 1;
    hsize_t dims[1] = {len};
    if (memtype_id == -1) {
        throw std::runtime_error("Write data set called with void pointer but no type info passed.");
    }
    write_dataset_nd(name, rank, dims, data,
        count, start,
        memtype_id, filetype_id,
        flag_parallel, flag_first_dim_parallel, flag_hyperslab, flag_collective);
}

///\todo need to update this to general mpi hyper slab selection
void H5OutputFile::write_dataset_nd(std::string name, int rank, hsize_t *dims, void *data,
    const std::vector<hsize_t>& count, const std::vector<hsize_t>& start,
    hid_t memtype_id, hid_t filetype_id,
    bool flag_parallel, bool flag_first_dim_parallel,
    bool flag_hyperslab, bool flag_collective)
{
#ifdef USEPARALLELHDF
    MPI_Comm comm = mpi_comm_write;
    MPI_Info info = MPI_INFO_NULL;
#endif
    hid_t dspace_id, dset_id, prop_id, memspace_id, ret;
    std::vector<hsize_t> chunks(rank);
    // Get HDF5 data type of the array in memory
    if (memtype_id == -1) {
        throw std::runtime_error("Write data set called with void pointer but no type info passed.");
    }
    // Determine type of the dataset to create
    if(filetype_id < 0) filetype_id = memtype_id;

#ifdef USEPARALLELHDF
    std::vector<unsigned long long> mpi_hdf_dims(rank*NProcsWrite), mpi_hdf_dims_tot(rank), dims_single(rank), dims_offset(rank);
    // _set_mpi_dim_and_offset(comm, rank, dims, dims_single, dims_offset, mpi_hdf_dims, mpi_hdf_dims_tot, flag_parallel, flag_first_dim_parallel);
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
    unsigned int large_dataset = 0;
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
// #ifdef USEPARALLELHDF
//     _set_mpi_hyperslab(dspace_id, memspace_id, rank, dims, mpi_hdf_dims_tot, flag_parallel, flag_hyperslab);
// #else
//     dspace_id = H5Screate_simple(rank, dims, NULL);
//     memspace_id = dspace_id;
// #endif

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
    bool iwrite = (dims[0] > 0);

    //set hyperslab
    if (!count.empty() && !start.empty()) {
        herr_t ret = H5Sselect_hyperslab(dspace_id, H5S_SELECT_SET, start.data(), NULL, count.data(), NULL);
    }

#ifdef USEPARALLELHDF
    // _set_mpi_dataset_properties(prop_id, iwrite,
    //     dspace_id, memspace_id,
    //     rank, dims, dims_offset,
    //     flag_parallel, flag_collective, flag_hyperslab);
#endif
    if (iwrite) {
        ret = H5Dwrite(dset_id, memtype_id, memspace_id, dspace_id, prop_id, data);
        if (ret < 0) io_error(std::string("Failed to write dataset: ")+name);
    }

    // Clean up (note that dtype_id is NOT a new object so don't need to close it)
    H5Pclose(prop_id);
#ifdef USEPARALLELHDF
    // if (flag_hyperslab && flag_parallel) H5Sclose(memspace_id);
#endif
    H5Sclose(dspace_id);
    H5Dclose(dset_id);
}
