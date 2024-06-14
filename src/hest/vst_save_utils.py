#####
# Utils for h5 file saving
####

import h5py


def initsave_hdf5(output_fpath, asset_dict, attr_dict= None, mode='a', static_shape=None, chunk_as_max_shape=False, verbose=0):
    with h5py.File(output_fpath, mode) as f:
        for key, val in asset_dict.items():
            data_shape = val.shape

            # if len(data_shape) == 1:
            #     val = np.expand_dims(val, axis=1)
            #     data_shape = val.shape

            if key not in f:
                data_type = val.dtype

                if data_type.kind == 'U':   # This is for catching numpy array of unicode strings
                    chunk_shape = (1, 1)
                    max_shape = (None, 1)
                    data_type = h5py.string_dtype(encoding='utf-8')         
                else:
                    if chunk_as_max_shape:
                        chunk_shape = data_shape
                        max_shape = data_shape
                    elif static_shape is None:
                        chunk_shape = (1,) + data_shape[1:]
                        max_shape = (None,) + data_shape[1:]
                    # else:
                    #     chunk_shape = static_shape
                    #     max_shape = static_shape

                if verbose:
                    print(key, data_shape, chunk_shape, max_shape)

                dset = f.create_dataset(key, 
                                        shape=data_shape, 
                                        maxshape=max_shape, 
                                        chunks=chunk_shape, 
                                        dtype=data_type)
                    
                dset[:] = val

                ### Save attribute dictionary
                if attr_dict is not None:
                    if key in attr_dict.keys():
                        for attr_key, attr_val in attr_dict[key].items():
                            dset.attrs[attr_key] = attr_val
            else:
                dset = f[key]
                dset.resize(len(dset) + data_shape[0], axis=0)
                #assert dset.dtype == val.dtype
                dset[-data_shape[0]:] = val
                
    return output_fpath