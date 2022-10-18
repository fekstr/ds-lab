import urllib
import urllib.request 

baseUrl = 'https://api.gdc.cancer.gov'

with open('gdc_manifest.2022-10-18.txt') as f:
    for line in f.readlines()[1:]:
        id, name = line.split('\t')[:2]
        urllib.request.urlretrieve(f'{baseUrl}/data/{id}', f'./slides/{name}')
        break # Add break so if you accidentally run this you won't dowload the entire dataset. Adjust this code as needed
