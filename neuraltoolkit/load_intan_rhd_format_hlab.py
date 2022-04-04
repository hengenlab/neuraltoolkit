# flake8: noqa
#! /bin/env python
#
# Modified for use in Hlab
# Michael Gibson 17 July 2015
# Modified Adrian Foy Sep 2018

#import sys, struct, math, os, time
#import numpy as np


#from intanutil.read_header import read_header
#from intanutil.get_bytes_per_data_block import get_bytes_per_data_block
#from intanutil.read_one_data_block import read_one_data_block
#from intanutil.notch_filter import notch_filter
#from intanutil.data_to_result import data_to_result

# qstring
##########################
def read_qstring(fid):
    import sys, struct, math, os, time
    import numpy as np

    """Read Qt style QString.  

    The first 32-bit unsigned number indicates the length of the string (in bytes).  
    If this number equals 0xFFFFFFFF, the string is null.

    Strings are stored as unicode.
    """

    length, = struct.unpack('<I', fid.read(4))
    if length == int('ffffffff', 16): return ""

    if length > (os.fstat(fid.fileno()).st_size - fid.tell() + 1) :
        print(length)
        raise Exception('Length too long.')

    # convert length from bytes to 16-bit Unicode words
    length = int(length / 2)

    data = []
    for i in range(0, length):
        c, = struct.unpack('<H', fid.read(2))
        data.append(c)

    if sys.version_info >= (3,0):
        a = ''.join([chr(c) for c in data])
    else:
        a = ''.join([unichr(c) for c in data])
    
    return a
  
#if __name__ == '__main__':
#    a=read_qstring(open(sys.argv[1], 'rb'))
#    print(a)
##########################

# data_to_result
##########################
def data_to_result(header, data, data_present):
    import sys, struct, math, os, time
    import numpy as np

    """Moves the header and data (if present) into a common object."""
    
    result = {}
    if header['num_amplifier_channels'] > 0 and data_present:
        result['t_amplifier'] = data['t_amplifier']
    if header['num_aux_input_channels'] > 0 and data_present:
        result['t_aux_input'] = data['t_aux_input']
    if header['num_supply_voltage_channels'] > 0 and data_present:
        result['t_supply_voltage'] = data['t_supply_voltage']
    if header['num_board_adc_channels'] > 0 and data_present:
        result['t_board_adc'] = data['t_board_adc']
    if (header['num_board_dig_in_channels'] > 0 or header['num_board_dig_out_channels'] > 0) and data_present:
        result['t_dig'] = data['t_dig']
    if header['num_temp_sensor_channels'] > 0 and data_present:
        result['t_temp_sensor'] = data['t_temp_sensor']
        
    if header['num_amplifier_channels'] > 0:
        result['spike_triggers'] = header['spike_triggers']
        
    result['notes'] = header['notes']
    result['frequency_parameters'] = header['frequency_parameters']
    
    if header['version']['major'] > 1:
        result['reference_channel'] = header['reference_channel']
    
    if header['num_amplifier_channels'] > 0:
        result['amplifier_channels'] = header['amplifier_channels']
        if data_present:
            result['amplifier_data'] = data['amplifier_data']
            
    if header['num_aux_input_channels'] > 0:
        result['aux_input_channels'] = header['aux_input_channels']
        if data_present:
            result['aux_input_data'] = data['aux_input_data']
            
    if header['num_supply_voltage_channels'] > 0:
        result['supply_voltage_channels'] = header['supply_voltage_channels']
        if data_present:
            result['supply_voltage_data'] = data['supply_voltage_data']
    
    if header['num_board_adc_channels'] > 0:
        result['board_adc_channels'] = header['board_adc_channels']
        if data_present:
            result['board_adc_data'] = data['board_adc_data']
            
    if header['num_board_dig_in_channels'] > 0:
        result['board_dig_in_channels'] = header['board_dig_in_channels']
        if data_present:
            result['board_dig_in_data'] = data['board_dig_in_data']
            
    if header['num_board_dig_out_channels'] > 0:
        result['board_dig_out_channels'] = header['board_dig_out_channels']
        if data_present:
            result['board_dig_out_data'] = data['board_dig_out_data']
    
    return result
##########################

# notch_filter 
##########################
#import math
#import numpy as np

def notch_filter(input, fSample, fNotch, Bandwidth):
    import sys, struct, math, os, time
    import numpy as np

    """Implements a notch filter (e.g., for 50 or 60 Hz) on vector 'input'.

    fSample = sample rate of data (input Hz or Samples/sec)
    fNotch = filter notch frequency (input Hz)
    Bandwidth = notch 3-dB bandwidth (input Hz).  A bandwidth of 10 Hz is
    recommended for 50 or 60 Hz notch filters; narrower bandwidths lead to
    poor time-domain properties with an extended ringing response to
    transient disturbances.

    Example:  If neural data was sampled at 30 kSamples/sec
    and you wish to implement a 60 Hz notch filter:

    out = notch_filter(input, 30000, 60, 10);
    """

    tstep = 1.0/fSample
    Fc = fNotch*tstep

    L = len(input)

    # Calculate IIR filter parameters
    d = math.exp(-2.0*math.pi*(Bandwidth/2.0)*tstep)
    b = (1.0 + d*d) * math.cos(2.0*math.pi*Fc)
    a0 = 1.0
    a1 = -b
    a2 = d*d
    a = (1.0 + d*d)/2.0
    b0 = 1.0
    b1 = -2.0 * math.cos(2.0*math.pi*Fc)
    b2 = 1.0

    out = np.zeros(len(input))
    out[0] = input[0]
    out[1] = input[1]
    # (If filtering a continuous data stream, change out[0:1] to the
    #  previous final two values of out.)

    # Run filter
    for i in range(2,L):
        out[i] = (a*b2*input[i-2] + a*b1*input[i-1] + a*b0*input[i] - a2*out[i-2] - a1*out[i-1])/a0

    return out
##########################

# read_one_data_block
##########################
#import sys, struct
#import numpy as np

def read_one_data_block(data, header, indices, fid):
    import sys, struct, math, os, time
    import numpy as np

    """Reads one 60 or 128 sample data block from fid into data, at the location indicated by indices."""

    # In version 1.2, we moved from saving timestamps as unsigned
    # integers to signed integers to accommodate negative (adjusted)
    # timestamps for pretrigger data['
    if (header['version']['major'] == 1 and header['version']['minor'] >= 2) or (header['version']['major'] > 1):
        data['t_amplifier'][indices['amplifier']:(indices['amplifier'] + header['num_samples_per_data_block'])] = np.array(struct.unpack('<' + 'i' * header['num_samples_per_data_block'], fid.read(4 * header['num_samples_per_data_block'])))
    else:
        data['t_amplifier'][indices['amplifier']:(indices['amplifier'] + header['num_samples_per_data_block'])] = np.array(struct.unpack('<' + 'I' * header['num_samples_per_data_block'], fid.read(4 * header['num_samples_per_data_block'])))

    if header['num_amplifier_channels'] > 0:
        tmp = np.fromfile(fid, dtype='uint16', count= header['num_samples_per_data_block'] * header['num_amplifier_channels'])
        data['amplifier_data'][range(header['num_amplifier_channels']), (indices['amplifier']):(indices['amplifier']+ header['num_samples_per_data_block'])] = tmp.reshape(header['num_amplifier_channels'], header['num_samples_per_data_block'])

    if header['num_aux_input_channels'] > 0:
        tmp = np.fromfile(fid, dtype='uint16', count= int((header['num_samples_per_data_block'] / 4) * header['num_aux_input_channels']))
        data['aux_input_data'][range(header['num_aux_input_channels']), indices['aux_input']:int(indices['aux_input']+ (header['num_samples_per_data_block']/4))] = tmp.reshape(header['num_aux_input_channels'], int(header['num_samples_per_data_block']/4))

    if header['num_supply_voltage_channels'] > 0:
        tmp = np.fromfile(fid, dtype='uint16', count=1 * header['num_supply_voltage_channels'])
        data['supply_voltage_data'][range(header['num_supply_voltage_channels']), indices['supply_voltage']:(indices['supply_voltage']+1)] = tmp.reshape(header['num_supply_voltage_channels'], 1)

    if header['num_temp_sensor_channels'] > 0:
        tmp = np.fromfile(fid, dtype='uint16', count=1 * header['num_temp_sensor_channels'])
        data['temp_sensor_data'][range(header['num_temp_sensor_channels']), indices['supply_voltage']:(indices['supply_voltage']+1)] = tmp.reshape(header['num_temp_sensor_channels'], 1)

    if header['num_board_adc_channels'] > 0:
        tmp = np.fromfile(fid, dtype='uint16', count= (header['num_samples_per_data_block']) * header['num_board_adc_channels'])
        data['board_adc_data'][range(header['num_board_adc_channels']), indices['board_adc']:(indices['board_adc']+ header['num_samples_per_data_block'])] = tmp.reshape(header['num_board_adc_channels'], header['num_samples_per_data_block'])

    if header['num_board_dig_in_channels'] > 0:
        data['board_dig_in_raw'][indices['board_dig_in']:(indices['board_dig_in']+ header['num_samples_per_data_block'])] = np.array(struct.unpack('<' + 'H' * header['num_samples_per_data_block'], fid.read(2 * header['num_samples_per_data_block'])))

    if header['num_board_dig_out_channels'] > 0:
        data['board_dig_out_raw'][indices['board_dig_out']:(indices['board_dig_out']+ header['num_samples_per_data_block'])] = np.array(struct.unpack('<' + 'H' * header['num_samples_per_data_block'], fid.read(2 * header['num_samples_per_data_block'])))
##########################

# get_bytes_per_data_block
##########################
def get_bytes_per_data_block(header):
    import sys, struct, math, os, time
    import numpy as np

    """Calculates the number of bytes in each 60 or 128 sample datablock."""

    # Each data block contains 60 or 128 amplifier samples.
    bytes_per_block = header['num_samples_per_data_block'] * 4  # timestamp data
    bytes_per_block = bytes_per_block + header['num_samples_per_data_block'] * 2 * header['num_amplifier_channels']

    # Auxiliary inputs are sampled 4x slower than amplifiers
    bytes_per_block = bytes_per_block + (header['num_samples_per_data_block'] / 4) * 2 * header['num_aux_input_channels']

    # Supply voltage is sampled 60 or 128x slower than amplifiers
    bytes_per_block = bytes_per_block + 1 * 2 * header['num_supply_voltage_channels']

    # Board analog inputs are sampled at same rate as amplifiers
    bytes_per_block = bytes_per_block + header['num_samples_per_data_block'] * 2 * header['num_board_adc_channels']

    # Board digital inputs are sampled at same rate as amplifiers
    if header['num_board_dig_in_channels'] > 0:
        bytes_per_block = bytes_per_block + header['num_samples_per_data_block'] * 2

    # Board digital outputs are sampled at same rate as amplifiers
    if header['num_board_dig_out_channels'] > 0:
        bytes_per_block = bytes_per_block + header['num_samples_per_data_block'] * 2

    # Temp sensor is sampled 60 or 128x slower than amplifiers
    if header['num_temp_sensor_channels'] > 0:
        bytes_per_block = bytes_per_block + 1 * 2 * header['num_temp_sensor_channels']

    return bytes_per_block
##########################

# read_header
##########################
#import sys, struct
#from intanutil.qstring import read_qstring

def read_header(fid):
    import sys, struct, math, os, time
    import numpy as np

    """Reads the Intan File Format header from the given file."""

    # Check 'magic number' at beginning of file to make sure this is an Intan
    # Technologies RHD2000 data file.
    magic_number, = struct.unpack('<I', fid.read(4)) 
    if magic_number != int('c6912702', 16): raise Exception('Unrecognized file type.')

    header = {}
    # Read version number.
    version = {}
    (version['major'], version['minor']) = struct.unpack('<hh', fid.read(4)) 
    header['version'] = version

    print('')
    print('Reading Intan Technologies RHD2000 Data File, Version {}.{}'.format(version['major'], version['minor']))
    print('')

    freq = {}

    # Read information of sampling rate and amplifier frequency settings.
    header['sample_rate'], = struct.unpack('<f', fid.read(4))
    (freq['dsp_enabled'], freq['actual_dsp_cutoff_frequency'], freq['actual_lower_bandwidth'], freq['actual_upper_bandwidth'], 
    freq['desired_dsp_cutoff_frequency'], freq['desired_lower_bandwidth'], freq['desired_upper_bandwidth']) = struct.unpack('<hffffff', fid.read(26))


    # This tells us if a software 50/60 Hz notch filter was enabled during
    # the data acquisition.
    notch_filter_mode, = struct.unpack('<h', fid.read(2))
    header['notch_filter_frequency'] = 0
    if notch_filter_mode == 1:
        header['notch_filter_frequency'] = 50
    elif notch_filter_mode == 2:
        header['notch_filter_frequency'] = 60
    freq['notch_filter_frequency'] = header['notch_filter_frequency']

    (freq['desired_impedance_test_frequency'], freq['actual_impedance_test_frequency']) = struct.unpack('<ff', fid.read(8))

    note1 = read_qstring(fid)
    note2 = read_qstring(fid)
    note3 = read_qstring(fid)
    header['notes'] = { 'note1' : note1, 'note2' : note2, 'note3' : note3}

    # If data file is from GUI v1.1 or later, see if temperature sensor data was saved.
    header['num_temp_sensor_channels'] = 0
    if (version['major'] == 1 and version['minor'] >= 1) or (version['major'] > 1) :
        header['num_temp_sensor_channels'], = struct.unpack('<h', fid.read(2))
        
    # If data file is from GUI v1.3 or later, load eval board mode.
    header['eval_board_mode'] = 0
    if ((version['major'] == 1) and (version['minor'] >= 3)) or (version['major'] > 1) :
        header['eval_board_mode'], = struct.unpack('<h', fid.read(2))
        
        
    header['num_samples_per_data_block'] = 60
    # If data file is from v2.0 or later (Intan Recording Controller), load name of digital reference channel
    if (version['major'] > 1):
        header['reference_channel'] = read_qstring(fid)
        header['num_samples_per_data_block'] = 128

    # Place frequency-related information in data structure. (Note: much of this structure is set above)
    freq['amplifier_sample_rate'] = header['sample_rate']
    freq['aux_input_sample_rate'] = header['sample_rate'] / 4
    freq['supply_voltage_sample_rate'] = header['sample_rate'] / header['num_samples_per_data_block']
    freq['board_adc_sample_rate'] = header['sample_rate']
    freq['board_dig_in_sample_rate'] = header['sample_rate']

    header['frequency_parameters'] = freq

    # Create structure arrays for each type of data channel.
    header['spike_triggers'] = []
    header['amplifier_channels'] = []
    header['aux_input_channels'] = []
    header['supply_voltage_channels'] = []
    header['board_adc_channels'] = []
    header['board_dig_in_channels'] = []
    header['board_dig_out_channels'] = []

    # Read signal summary from data file header.

    number_of_signal_groups, = struct.unpack('<h', fid.read(2))
    print('n signal groups {}'.format(number_of_signal_groups))

    for signal_group in range(1, number_of_signal_groups + 1):
        signal_group_name = read_qstring(fid)
        signal_group_prefix = read_qstring(fid)
        (signal_group_enabled, signal_group_num_channels, signal_group_num_amp_channels) = struct.unpack('<hhh', fid.read(6))

        if (signal_group_num_channels > 0) and (signal_group_enabled > 0):
            for signal_channel in range(0, signal_group_num_channels):
                new_channel = {'port_name' : signal_group_name, 'port_prefix' : signal_group_prefix, 'port_number' : signal_group}
                new_channel['native_channel_name'] = read_qstring(fid)
                new_channel['custom_channel_name'] = read_qstring(fid)
                (new_channel['native_order'], new_channel['custom_order'], signal_type, channel_enabled, new_channel['chip_channel'], new_channel['board_stream']) = struct.unpack('<hhhhhh', fid.read(12))
                new_trigger_channel = {}
                (new_trigger_channel['voltage_trigger_mode'], new_trigger_channel['voltage_threshold'], new_trigger_channel['digital_trigger_channel'], new_trigger_channel['digital_edge_polarity'])  = struct.unpack('<hhhh', fid.read(8))
                (new_channel['electrode_impedance_magnitude'], new_channel['electrode_impedance_phase']) = struct.unpack('<ff', fid.read(8))

                if channel_enabled:
                    if signal_type == 0:
                        header['amplifier_channels'].append(new_channel)
                        header['spike_triggers'].append(new_trigger_channel)
                    elif signal_type == 1:
                        header['aux_input_channels'].append(new_channel)
                    elif signal_type == 2:
                        header['supply_voltage_channels'].append(new_channel)
                    elif signal_type == 3:
                        header['board_adc_channels'].append(new_channel)
                    elif signal_type == 4:
                        header['board_dig_in_channels'].append(new_channel)
                    elif signal_type == 5:
                        header['board_dig_out_channels'].append(new_channel)
                    else:
                        raise Exception('Unknown channel type.')
                        
    # Summarize contents of data file.
    header['num_amplifier_channels'] = len(header['amplifier_channels'])
    header['num_aux_input_channels'] = len(header['aux_input_channels'])
    header['num_supply_voltage_channels'] = len(header['supply_voltage_channels'])
    header['num_board_adc_channels'] = len(header['board_adc_channels'])
    header['num_board_dig_in_channels'] = len(header['board_dig_in_channels'])
    header['num_board_dig_out_channels'] = len(header['board_dig_out_channels'])

    return header

#if __name__ == '__main__':
#    h=read_header(open(sys.argv[1], 'rb'))
#    print(h)
##########################

def read_data(filename):
    import sys, struct, math, os, time
    import numpy as np

    """Reads Intan Technologies RHD2000 data file generated by evaluation board GUI.
    
    Data are returned in a dictionary, for future extensibility.
    """

    tic = time.time()
    fid = open(filename, 'rb')
    filesize = os.path.getsize(filename)

    header = read_header(fid)

    print('Found {} amplifier channel{}.'.format(header['num_amplifier_channels'], plural(header['num_amplifier_channels'])))
    print('Found {} auxiliary input channel{}.'.format(header['num_aux_input_channels'], plural(header['num_aux_input_channels'])))
    print('Found {} supply voltage channel{}.'.format(header['num_supply_voltage_channels'], plural(header['num_supply_voltage_channels'])))
    print('Found {} board ADC channel{}.'.format(header['num_board_adc_channels'], plural(header['num_board_adc_channels'])))
    print('Found {} board digital input channel{}.'.format(header['num_board_dig_in_channels'], plural(header['num_board_dig_in_channels'])))
    print('Found {} board digital output channel{}.'.format(header['num_board_dig_out_channels'], plural(header['num_board_dig_out_channels'])))
    print('Found {} temperature sensors channel{}.'.format(header['num_temp_sensor_channels'], plural(header['num_temp_sensor_channels'])))
    print('')

    # Determine how many samples the data file contains.
    bytes_per_block = get_bytes_per_data_block(header)

    # How many data blocks remain in this file?
    data_present = False
    bytes_remaining = filesize - fid.tell()
    if bytes_remaining > 0:
        data_present = True

    if bytes_remaining % bytes_per_block != 0:
        raise Exception('Something is wrong with file size : should have a whole number of data blocks')

    num_data_blocks = int(bytes_remaining / bytes_per_block)

    num_amplifier_samples = header['num_samples_per_data_block'] * num_data_blocks
    num_aux_input_samples = int((header['num_samples_per_data_block'] / 4) * num_data_blocks)
    num_supply_voltage_samples = 1 * num_data_blocks
    num_board_adc_samples = header['num_samples_per_data_block'] * num_data_blocks
    num_board_dig_in_samples = header['num_samples_per_data_block'] * num_data_blocks
    num_board_dig_out_samples = header['num_samples_per_data_block'] * num_data_blocks

    record_time = num_amplifier_samples / header['sample_rate']

    if data_present:
        print('File contains {:0.3f} seconds of data.  Amplifiers were sampled at {:0.2f} kS/s.'.format(record_time, header['sample_rate'] / 1000))
    else:
        print('Header file contains no data.  Amplifiers were sampled at {:0.2f} kS/s.'.format(header['sample_rate'] / 1000))

    if data_present:
        # Pre-allocate memory for data.
        print('')
        print('Allocating memory for data...')

        data = {}
        if (header['version']['major'] == 1 and header['version']['minor'] >= 2) or (header['version']['major'] > 1):
            data['t_amplifier'] = np.zeros(num_amplifier_samples, dtype=np.int)
        else:
            data['t_amplifier'] = np.zeros(num_amplifier_samples, dtype=np.uint)

        data['amplifier_data'] = np.zeros([header['num_amplifier_channels'], num_amplifier_samples], dtype=np.uint)
        data['aux_input_data'] = np.zeros([header['num_aux_input_channels'], num_aux_input_samples], dtype=np.uint)
        data['supply_voltage_data'] = np.zeros([header['num_supply_voltage_channels'], num_supply_voltage_samples], dtype=np.uint)
        data['temp_sensor_data'] = np.zeros([header['num_temp_sensor_channels'], num_supply_voltage_samples], dtype=np.uint)
        data['board_adc_data'] = np.zeros([header['num_board_adc_channels'], num_board_adc_samples], dtype=np.uint)
        
        # by default, this script interprets digital events (digital inputs and outputs) as booleans
        # if unsigned int values are preferred(0 for False, 1 for True), replace the 'dtype=np.bool' argument with 'dtype=np.uint' as shown
        # the commented line below illustrates this for digital input data; the same can be done for digital out
        
        #data['board_dig_in_data'] = np.zeros([header['num_board_dig_in_channels'], num_board_dig_in_samples], dtype=np.uint)
        data['board_dig_in_data'] = np.zeros([header['num_board_dig_in_channels'], num_board_dig_in_samples], dtype=np.bool)
        data['board_dig_in_raw'] = np.zeros(num_board_dig_in_samples, dtype=np.uint)
        
        data['board_dig_out_data'] = np.zeros([header['num_board_dig_out_channels'], num_board_dig_out_samples], dtype=np.bool)
        data['board_dig_out_raw'] = np.zeros(num_board_dig_out_samples, dtype=np.uint)

        # Read sampled data from file.
        print('Reading data from file...')

        # Initialize indices used in looping
        indices = {}
        indices['amplifier'] = 0
        indices['aux_input'] = 0
        indices['supply_voltage'] = 0
        indices['board_adc'] = 0
        indices['board_dig_in'] = 0
        indices['board_dig_out'] = 0

        print_increment = 10
        percent_done = print_increment
        for i in range(num_data_blocks):
            read_one_data_block(data, header, indices, fid)

            # Increment indices
            indices['amplifier'] += header['num_samples_per_data_block']
            indices['aux_input'] += int(header['num_samples_per_data_block'] / 4)
            indices['supply_voltage'] += 1
            indices['board_adc'] += header['num_samples_per_data_block']
            indices['board_dig_in'] += header['num_samples_per_data_block']
            indices['board_dig_out'] += header['num_samples_per_data_block']            

            fraction_done = 100 * (1.0 * i / num_data_blocks)
            if fraction_done >= percent_done:
                print('{}% done...'.format(percent_done))
                percent_done = percent_done + print_increment

        # Make sure we have read exactly the right amount of data.
        bytes_remaining = filesize - fid.tell()
        if bytes_remaining != 0: raise Exception('Error: End of file not reached.')



    # Close data file.
    fid.close()

    if (data_present):
        print('Parsing data...')

        # Extract digital input channels to separate variables.
        for i in range(header['num_board_dig_in_channels']):
            data['board_dig_in_data'][i, :] = np.not_equal(np.bitwise_and(data['board_dig_in_raw'], (1 << header['board_dig_in_channels'][i]['native_order'])), 0)

        # Extract digital output channels to separate variables.
        for i in range(header['num_board_dig_out_channels']):
            data['board_dig_out_data'][i, :] = np.not_equal(np.bitwise_and(data['board_dig_out_raw'], (1 << header['board_dig_out_channels'][i]['native_order'])), 0)

        # Scale voltage levels appropriately.
        data['amplifier_data'] = np.multiply(0.195, (data['amplifier_data'].astype(np.int32) - 32768))      # units = microvolts
        data['aux_input_data'] = np.multiply(37.4e-6, data['aux_input_data'])               # units = volts
        data['supply_voltage_data'] = np.multiply(74.8e-6, data['supply_voltage_data'])     # units = volts
        if header['eval_board_mode'] == 1:
            data['board_adc_data'] = np.multiply(152.59e-6, (data['board_adc_data'].astype(np.int32) - 32768)) # units = volts
        elif header['eval_board_mode'] == 13:
            data['board_adc_data'] = np.multiply(312.5e-6, (data['board_adc_data'].astype(np.int32) - 32768)) # units = volts
        else:
            data['board_adc_data'] = np.multiply(50.354e-6, data['board_adc_data'])           # units = volts
        data['temp_sensor_data'] = np.multiply(0.01, data['temp_sensor_data'])               # units = deg C

        # Check for gaps in timestamps.
        num_gaps = np.sum(np.not_equal(data['t_amplifier'][1:]-data['t_amplifier'][:-1], 1))
        if num_gaps == 0:
            print('No missing timestamps in data.')
        else:
            print('Warning: {0} gaps in timestamp data found.  Time scale will not be uniform!'.format(num_gaps))

        # Scale time steps (units = seconds).
        data['t_amplifier'] = data['t_amplifier'] / header['sample_rate']
        data['t_aux_input'] = data['t_amplifier'][range(0, len(data['t_amplifier']), 4)]
        data['t_supply_voltage'] = data['t_amplifier'][range(0, len(data['t_amplifier']), header['num_samples_per_data_block'])]
        data['t_board_adc'] = data['t_amplifier']
        data['t_dig'] = data['t_amplifier']
        data['t_temp_sensor'] = data['t_supply_voltage']

        # If the software notch filter was selected during the recording, apply the
        # same notch filter to amplifier data here.
        if header['notch_filter_frequency'] > 0:
            print('Applying notch filter...')

            print_increment = 10
            percent_done = print_increment
            for i in range(header['num_amplifier_channels']):
                data['amplifier_data'][i,:] = notch_filter(data['amplifier_data'][i,:], header['sample_rate'], header['notch_filter_frequency'], 10)

                fraction_done = 100 * (i / header['num_amplifier_channels'])
                if fraction_done >= percent_done:
                    print('{}% done...'.format(percent_done))
                    percent_done += print_increment
    else:
        data = [];

    # Move variables to result struct.
    result = data_to_result(header, data, data_present)

    print('Done!  Elapsed time: {0:0.1f} seconds'.format(time.time() - tic))
    return result

def plural(n):
    import sys, struct, math, os, time
    import numpy as np

    """Utility function to optionally pluralize words based on the value of n.
    """

    if n == 1:
        return ''
    else:
        return 's'

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('TkAgg')
    import sys, struct, math, os, time
    import numpy as np
    from tkinter import Tk
    from tkinter.filedialog import askopenfilename
    import sys, struct, math, os, time
    import numpy as np


    if len(sys.argv) > 1:
        a=read_data(sys.argv[1])
    else:
        Tk().withdraw() 
        filename = askopenfilename() # open file browser
        print(filename)
        a=read_data(filename)

    t = np.array(a['t_amplifier'][0])
    data = np.array(a['amplifier_data'])

    # channel map
    chanMap = np.array([25, 26, 27, 28, 29, 30, 31, 32, 1, 2, 3, 4, 5, 6, 7, 8, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9]) - 1
    data_chmap = data[chanMap, :]

    print('Time is %f', t)
    print(data_chmap)
