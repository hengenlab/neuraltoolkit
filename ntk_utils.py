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
            elist.append(re.findall(r'\S+@\S+', sline))
    email_list = list(filter(None, elist))
    return email_list
