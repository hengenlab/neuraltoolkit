import numpy as np
import time
import matplotlib.pyplot as plt


def remove_large_noise(ddgc_filt, max_value_to_check=3000,
                       windowval=500, checkchans=None,
                       lplot=0):

    '''
    remove_large_noise(ddgc_filt, max_value_to_check=2500,
                       windowval=500, checkchans=None,
                       lplot=0)

    ddgc_filt: raw data matrix
    max_value_to_check : max value to remove (default 2500)
    windowval : window to zero (default 500)
    checkchans : list of 4 channels (default None)
    lplot : 1 to plot first channel from checkchans (default 0)

    data: cleaned up ddgc_filt

    raise ValueError
    '''

    if (len(checkchans) != 4):
        raise ValueError(f'Error: checkchans length not 4: {len(checkchans)}')

    ddgc = ddgc_filt * 1

    # max_value_to_check = 2500
    # width_to_check = 1
    # windowval = 500

    # Artifact checks
    otic = time.time()
    edges = None
    edges1 = None
    edges2 = None
    edges3 = None
    edges4 = None
    edges1 = np.argwhere(np.abs(ddgc[checkchans[0], :]) >
                         max_value_to_check).flatten()
    edges2 = np.argwhere(np.abs(ddgc[checkchans[1], :]) >
                         max_value_to_check).flatten()
    edges3 = np.argwhere(np.abs(ddgc[checkchans[2], :]) >
                         max_value_to_check).flatten()
    edges4 = np.argwhere(np.abs(ddgc[checkchans[3], :]) >
                         max_value_to_check).flatten()
    edges = np.unique(np.concatenate((edges1, edges2, edges3, edges4)))
    print(f'shape edges {edges.shape}', flush=True)

    for i in edges:
        # print("iii " ,i,  " ", min(i, (i - windowval)),  " ",
        #       min(ddgc.shape[1], (i + windowval)))
        ddgc[:,
             min(i, (i - windowval)): min(ddgc.shape[1], (i + windowval))] = 0

    otoc = time.time()
    print(f'Artifact removal took {otoc - otic} seconds')

    if lplot:
        fig, ax = plt.subplots(nrows=3, figsize=(40, 10), sharex=True)
        fig.suptitle(f'Channel {checkchans[0]}')
        for i in checkchans[0:1]:
            ax[3*i].plot(ddgc_filt[i, :])
            ax[(3*i) + 1].plot(ddgc[i, :])
            ax[(3*i) + 2].plot(edges, np.arange(len(edges)))
        plt.show()

    return ddgc


def get_tetrode_channels_from_channelnum(channel1, ch_grp_size=4):
    '''
    get_tetrode_channels_from_channelnum(channel1, ch_grp_size=4)

    channel1 : channel number in tetrode
    ch_grp_size : default 4

    returns list of channels in that tetrode
    '''

    return np.arange(channel1 - (channel1 % ch_grp_size),
                     channel1 + (ch_grp_size -
                     (channel1 % ch_grp_size))).tolist()
