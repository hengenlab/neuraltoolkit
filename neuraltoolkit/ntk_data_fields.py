import requests
import json


def get_data_fields(animal=None, field=None, probenum=None, region=None):

    '''
    animal : animal name (5)
    field : field to return
            field list : 'animal_name', 'species',
                         'strain', 'genotype',
                         'sex', 'animal_dob',
                         'num_chan', 'num_regions',
                         'num_probes',
                         'implant_date', 'surgeon',
                         'electrode',  'headstage',
                         'daqsys', 'sac_date',
                         'probe_num', 'chanmap',
                         'region', 'chan_range',
                         'location'

                         location order is [AP, ML, DV]

                         'all' returns everything as json file

    probenum : if field is not 'all' give either probenum or region
    region :   if field is not 'all' give either probenum or region


    raises ValueError
    '''

    animal = str(animal).lower()
    field = str(field).lower()
    region = str(region).upper()
    # print(f'field {field}')
    if ((field != 'all') and ((probenum is None) and (region is None))):
        raise ValueError(f'Check probenum {probenum} and region {region}')

    response_API = \
        requests.get(f'http://mousecache.wustl.edu/name?name={animal}')
    if response_API.status_code == 200:
        data = response_API.text
        data_animal_json = json.loads(data)
        if data_animal_json == []:
            raise ValueError(f'Check animal {animal}')

        if ((probenum is not None) and (field != 'all')):
            probe_num = [indx for indx, i in enumerate(data_animal_json)
                         if i['probe_num'] == probenum][0]
        elif ((region is not None) and (field != 'all')):
            probe_num = [indx for indx, i in enumerate(data_animal_json)
                         if i['region'] == region][0]

        if field == 'all':
            print(data_animal_json)
            return data_animal_json

        elif field == 'animal_name':
            print(f"Animal {data_animal_json[probe_num]['animal_name']}")
            return data_animal_json[probe_num]['animal_name']
        elif field == 'species':
            print(f"Species {data_animal_json[probe_num]['species']}")
            return data_animal_json[probe_num]['species']
        elif field == 'strain':
            print(f"Strain {data_animal_json[probe_num]['strain']}")
            return data_animal_json[probe_num]['strain']
        elif field == 'genotype':
            print(f"Genotype {data_animal_json[probe_num]['genotype']}")
            return data_animal_json[probe_num]['genotype']
        elif field == 'sex':
            print(f"Sex {data_animal_json[probe_num]['sex']}")
            return data_animal_json[probe_num]['sex']
        elif field == 'animal_dob':
            print(f"DOB {data_animal_json[probe_num]['animal_dob']}")
            return data_animal_json[probe_num]['animal_dob']
        elif field == 'num_chan':
            print(f"Num channels {data_animal_json[probe_num]['num_chan']}")
            return data_animal_json[probe_num]['num_chan']
        elif field == 'num_regions':
            print(f"Num regions {data_animal_json[probe_num]['num_regions']}")
            return data_animal_json[probe_num]['num_regions']
        elif field == 'num_probes':
            print(f"Num_probes {data_animal_json[probe_num]['num_probes']}")
            return data_animal_json[probe_num]['num_probes']
        elif field == 'implant_date':
            print("implant_date ")
            print(f"{data_animal_json[probe_num]['implant_date']}")
            return data_animal_json[probe_num]['implant_date']
        elif field == 'surgeon':
            print(f"surgeon {data_animal_json[probe_num]['surgeon']}")
            return data_animal_json[probe_num]['surgeon']
        elif field == 'electrode':
            print(f"Electrode {data_animal_json[probe_num]['electrode']}")
            return data_animal_json[probe_num]['electrode']
        elif field == 'headstage':
            print(f"Headstage {data_animal_json[probe_num]['headstage']}")
            return data_animal_json[probe_num]['headstage']
        elif field == 'daqsys':
            print(f"Daqsys {data_animal_json[probe_num]['daqsys']}")
            return data_animal_json[probe_num]['daqsys']
        elif field == 'sac_date':
            print(f"Sac_date {data_animal_json[probe_num]['sac_date']}")
            return data_animal_json[probe_num]['sac_date']
        elif field == 'chanmap':
            print(f"Chanmap {data_animal_json[probe_num]['chanmap']}")
            return data_animal_json[probe_num]['chanmap']
        elif field == 'region':
            print(f"Region {data_animal_json[probe_num]['region']}")
            return data_animal_json[probe_num]['region']
        elif field == 'chan_range':
            print(f"Chanrange {data_animal_json[probe_num]['chan_range']}")
            return data_animal_json[probe_num]['chan_range']
        elif field == 'location':
            print('Location')
            print(f"\tAP {data_animal_json[probe_num]['ap']}")
            print(f"\tML {data_animal_json[probe_num]['ml']}")
            print(f"\tDV {data_animal_json[probe_num]['dv']}")
            return [data_animal_json[probe_num]['ap'],
                    data_animal_json[probe_num]['ml'],
                    data_animal_json[probe_num]['dv']]

        elif field == 'probe_num':
            # if ((probenum == None) and (region != None)):
            print(f"Probe num { data_animal_json[probe_num]['probe_num']}")
            return data_animal_json[probe_num]['probe_num']
        elif field == 'region':
            # if ((probenum != None) and (region == None)):
            print(f"Region {data_animal_json[probe_num]['region']}")
            return data_animal_json[probe_num]['region']

        else:
            raise ValueError(f'Check field {field}')
    else:
        raise ValueError('Error')
