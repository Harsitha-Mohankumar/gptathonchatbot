import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import time
import random

# Define a helper function to handle requests with retries
def safe_get(url, retries=3, delay=3):
    for i in range(retries):
        try:
            response = requests.get(url.strip())
            response.raise_for_status()  # Ensure we raise an error for bad HTTP status codes
            return response
        except requests.exceptions.RequestException as e:
            print(f"Request failed for {url}, attempt {i+1} of {retries}. Error: {e}")
            time.sleep(delay * (i + 1))  # Exponential backoff
    return None  # Return None if retries are exhausted

# Function to extract data from the first link
def first_link_data(link):
    doctor_info = {}

    response = safe_get(link)
    if not response:
        return doctor_info  # Return empty data if the request fails

    soup = BeautifulSoup(response.text, 'html.parser')

    table_data = soup.find('div', class_='col-xs-6')
    if table_data:
        first_table_data = table_data.find('table')
        if first_table_data:
            rows = first_table_data.find_all('tr')
            doctor_info['Identifier'] = rows[0].find_all('td')[1].get_text(strip=True) if len(rows) > 0 else ''
            doctor_info['date'] = rows[1].find_all('td')[1].get_text(strip=True) if len(rows) > 1 else ''
            doctor_info['Shape'] = rows[2].find_all('td')[1].get_text(strip=True) if len(rows) > 2 else ''
            doctor_info['Place'] = rows[3].find_all('td')[1].get_text(strip=True) if len(rows) > 3 else ''
            doctor_info['Duration'] = rows[4].find_all('td')[1].get_text(strip=True) if len(rows) > 4 else ''

    table = soup.find_all('tbody')
    if table:
        list_for_table1 = []
        table1_rows = table[0].find_all('tr')
        for row in table1_rows:
            columns = row.find_all('td')
            if len(columns) == 4:
                list_for_table1.append({
                    'Full_name': columns[0].get_text(strip=True),
                    'Quality': columns[1].get_text(strip=True),
                    'Works_for': columns[2].get_text(strip=True),
                    'Represents': columns[3].get_text(strip=True)
                })
        doctor_info['Assistants'] = list_for_table1
        doctor_info['Subjects_covered'] = ", \n".join( [i.find('td').get_text(strip=True) for i in table[1].find_all('tr')]
                                                    ) if len(table) > 1 else ''
        
        doctor_info['Specification'] = table[2].find('tr').find('td').get_text(strip=True) if len(table) > 2 else ''

    return doctor_info

# Function to structuring data
def structuring_data(df):

    # Set the first row as column headers
    # df.columns = df.iloc[0]
    # df = df[1:]  # Skip the header row

    # Column containing the assistants' data
    assistants_column = 'Assistants'

    # List to store expanded rows
    expanded_rows = []

    if assistants_column in df.columns:
        for _, row in df.iterrows():
            assistants_data = row[assistants_column]

            # Parse the assistants' data
            assistants_list = assistants_data

            is_first_row = True

            for assistant in assistants_list:
                # Create a new row with the assistant's data
                new_row = row.copy()
                new_row['Assistant_Full_name'] = assistant.get('Full_name', None)
                new_row['Assistant_Quality'] = assistant.get('Quality', None)
                new_row['Assistant_Works_for'] = assistant.get('Works_for', None)
                new_row['Assistant_Represents'] = assistant.get('Represents', None)

                # For subsequent rows, set non-assistant columns to None
                if not is_first_row:
                    for col in row.index:
                        if col not in ['Assistant_Full_name', 'Assistant_Quality', 'Assistant_Works_for', 'Assistant_Represents']:
                            new_row[col] = None
                else:
                    is_first_row = False

                expanded_rows.append(new_row)

            # If no assistants, keep the original row with empty assistant data
            if not assistants_list:
                new_row = row.copy()
                print(f"error: {row['Assistants']}")
                new_row['Assistant_Full_name'] = None
                new_row['Assistant_Quality'] = None
                new_row['Assistant_Works_for'] = None
                new_row['Assistant_Represents'] = None
                expanded_rows.append(new_row)

    # Create a new DataFrame from the expanded rows
    expanded_df = pd.DataFrame(expanded_rows)

    # Ensure the new columns are in the desired order
    expanded_df = expanded_df[df.columns.tolist() + [
        'Assistant_Full_name',
        'Assistant_Quality',
        'Assistant_Works_for',
        'Assistant_Represents'
    ]]

    # Drop the original Assistants column
    expanded_df.drop(columns=[assistants_column], inplace=True)

    return expanded_df

# Main scraping process
entire_doctors_list = []

for i in range(1, 3):  # Iterate through pages 1 and 2
    page_url = f'https://www.leylobby.gob.cl/instituciones/AO001/audiencias/2024?page={i}'
    response = safe_get(page_url)
    if not response:
        continue  # Skip this page if request fails
    
    soup = BeautifulSoup(response.text, 'html.parser')
    tbody = soup.find('tbody')
    rows = tbody.find_all('tr') if tbody else []
    
    for row in tqdm(rows, desc=f"Processing page {i}"):
        columns = row.find_all('td')
        if columns:
            full_name = columns[0].get_text(strip=True)
            Position = columns[1].get_text(strip=True)
            link = columns[2].find('a')['href']
            
            # Get the link data with retries
            response = safe_get(link)
            if not response:
                continue  # Skip if the link fails to load
            
            soup = BeautifulSoup(response.text, 'html.parser')
            tbody = soup.find('tbody')
            sub_rows = tbody.find_all('tr') if tbody else []
            
            for sub_row in tqdm(sub_rows, desc="Processing subrows"):
                columns = sub_row.find_all('td')
                if len(columns) == 7:
                    link_data = columns[6].find('a')['href']
                    time.sleep(random.uniform(1, 2))  # Sleep for a random time between 1 and 2 seconds to avoid hitting the server too hard
                    
                    first_link_data_dist = first_link_data(link_data)
                    entire_doctors_list.append({
                        'full_name': full_name,
                        'Position': Position,
                        'link': link_data,
                        'Identifier': first_link_data_dist.get('Identifier', ''),
                        'date': first_link_data_dist.get('date', ''),
                        'Shape': first_link_data_dist.get('Shape', ''),
                        'Place': first_link_data_dist.get('Place', ''),
                        'Duration': first_link_data_dist.get('Duration', ''),
                        'Assistants': first_link_data_dist.get('Assistants', []),
                        'Subjects_covered': first_link_data_dist.get('Subjects_covered', ''),
                        'Specification': first_link_data_dist.get('Specification', '')
                    })
    
# Convert scraped data to DataFrame
df = pd.DataFrame(entire_doctors_list)

# save to csv
df.to_excel('output/leylobby_doctors_data.xlsx', index=False)

# Structuring data
df = structuring_data(df)

# Save DataFrame to CSV file
df.to_excel('output/leylobby_doctors_data_structured.xlsx', index=False)