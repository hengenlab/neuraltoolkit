#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Utility functions of neuraltoolkit

Hengen Lab
Washington University in St. Louis
Author: Kiran Bhaskaran-Nair
Email: kbn.git@gmail.com
Version:  0.1


List of functions/class in ntk_utils
extract_email_fromtxtfile(filename)
'''

import json
import os.path as op
import numpy as np


# Extract email from a txt file
def extract_email_fromtxtfile(filename):

    '''
    extract_email_fromtxtfile(filename)
    filename = email_list.txt
    contents of email_list.txt
        username1@email.com  # username1
        username2@email.com  # username2

    output email_list, with all emails in the file
    '''

    import re

    elist = []
    with open(filename) as emailfile:
        for line in emailfile:
            # remove '\n' at end of line
            sline = line.rstrip()
            elist.extend(re.findall(r'\S+@\S+', sline))
    email_list = list(filter(None, elist))
    return email_list


def check_ping_status(ip_system):

    '''
    Check whether a system is up using ping

    check_ping_status(ip_system)
    status = check_ping_status("127.0.0.1")
    ip_system : ip of the system

    return 1 if up or 0 if down
    '''

    from platform import system as system_name
    import subprocess

    if system_name().lower() == 'windows':
        command = str('ping -n 3 ') + str(ip_system)
    elif system_name().lower() == 'linux':
        command = str('ping -c 3 ') + str(ip_system)
    elif system_name().lower() == 'darwin':
        command = str('ping -c 3 ') + str(ip_system)
    try:
        _ = subprocess.check_output(command, shell=True)
        # print(output)
        return 1
    except subprocess.CalledProcessError as e:
        if 0:   # debug
            print(e)
        return 0


def file_to_dict(filename, k_dtype=str, v_dtype=int):

    '''
     Read contents of filename to dictionary (dict_new)

     contents of name_mobile.txt
       name1 1111
       name2 2222
     dict_sms = ntk.file_to_dict('name_mobile.txt', k_dtype=str, v_dtype=int)

     filename : name of file to read
     k_dtype : dictionary key data type default(int)
     v_dtype : dictionary valye data type default(str)

     return dict_new
    '''

    import os.path as op

    dict_new = {}
    try:
        op.isfile(filename)
    except OSError as e:
        print("Error: %s - %s." % (e.filename, e.strerror))

    with open(filename) as f2dict:
        for line in f2dict:
            (key, val) = line.split()
            dict_new[k_dtype(key)] = v_dtype(val)
    return dict_new


def send_sms(from_email, from_pass, to_email_list, msg):

    '''
    To send sms using email
    send_sms(from_email, from_pass, to_email_list, msg)

    from_email: Email from which emails for sms are send
    from_pass: Passwork of email from which emails for sms are send
    to_email_list: list of mobile numbers with appropriate email extension
                   for example, '1234567890@@mms.att.net'
    msg: message to send

    ntk.send_sms('username@gmail.com', 'password',
                 ['mobilenumber@mms.att.net'], 'ecubeissues')
    '''

    import smtplib

    # checks
    if not isinstance(from_email, str):
        raise ValueError('Argument {} not a string'.format(from_email))
    if not isinstance(from_pass, str):
        raise ValueError('Argument {} not a string'.format(from_pass))
    if not isinstance(msg, str):
        raise ValueError('Argument {} not a string'.format(msg))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()

    # Next, log in to the server
    # server.login("youremailusername", "password")
    server.login(from_email.split('@')[0], from_pass)

    # Send the mail
    if isinstance(to_email_list, str):
        server.sendmail(from_email, to_email_list, msg)
    elif isinstance(to_email_list, list):
        for to_email in to_email_list:
            if isinstance(to_email, str):
                server.sendmail(from_email, to_email, msg)
            else:
                raise ValueError('Argument {} is not a string or list'
                                 .format(to_email_list))

    else:
        raise ValueError('Argument {} is not a string or list'
                         .format(to_email_list))


def load_json_file(json_file, verbose=0):
    '''
    Load json file and return dictionary

    json_data = load_json_file(json_file, verbose=1)

    json_file : json file with path, /home/kbn/data.json
    verbose : default(0), if 1 prints json_data


    '''

    # check file exists
    if not (op.isfile(json_file) and op.exists(json_file)):
        raise FileNotFoundError(f'File {json_file} not found')

    with open(json_file, 'r') as f:
        json_data = json.load(f)
    if verbose:
        print(json_data)
    return json_data


def find_edges_from_consecutive(data, step=1, lverbose=0):

    '''
    find_edges_from_consecutive(data, step=1, lverbose=0)
    data : data of set of consecutive data
    step :  Spacing between values (default 1), tested only for 1
    lverbose : verbosity (default 0), 1 to be verbose

    return edges of consecutive data
    '''

    if step !=1:
        raise ValueError('Only tested for step=1')

    # important to sort array for this to work correctly
    data = np.sort(data)

    data_split = np.split(data, np.where(np.diff(data) != step)[0]+1)
    data_split = np.asarray(data_split, dtype='object')
    edges = None
    edges = []
    if lverbose:
        print(f'data_split sh {data_split.shape}')
    for indx, data_split_i in enumerate(data_split):
        if lverbose:
            print(f'data_split_i {data_split_i[0]} {data_split_i[-1]}')
        edges.append([data_split_i[0], data_split_i[-1]])

    return edges
