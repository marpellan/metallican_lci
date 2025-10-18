from brightway2 import Database
import pandas as pd
from constants import CA_provinces

class LCIDatabaseBuilder:
    """
    A class to build and populate Brightway2 LCI databases from DataFrames.
    """

    def __init__(self, db_name):
        """
        Initialize the builder with a database name.
        """
        self.db_name = db_name
        self.db = Database(db_name)
        self.db.register()
        self.lcis = {}  # Dictionary to hold activities before writing to the database

    def build_lci_entries(self, df, facility_col, site_id_col): # Method of the LCIDatabaseBuilder
        """
        Initialize LCI activities from a DataFrame.
        """
        self.lcis = {}
        for _, row in df.iterrows():
            name = f"{row['mining_processing_type']} {row['commodities']}, {row[facility_col]}".strip()
            location = CA_provinces.get(row['province'], 'CA')
            site_id = row[site_id_col]
            key = (self.db_name, name)
            self.lcis[key] = {
                'name': name,
                'unit': 'kilogram',
                'location': location,
                'exchanges': [],
                'type': 'process',
                'description': f"This is a site-specific LCI drawn from the MetalliCan database. Site ID is {site_id}."
            }
        return self.lcis

    def populate_technosphere_exchanges(self, technosphere_df, site_id_column='site_id'):
        """
        Populate technosphere exchanges for all activities.
        """
        for key, process in self.lcis.items():
            site_id = process['description'].split('Site ID is ')[-1].strip('. ')
            site_exchanges = technosphere_df[technosphere_df[site_id_column] == site_id]
            for _, row in site_exchanges.iterrows():
                exchange = {
                    'input': row['input_key'],
                    'amount': row['amount'],
                    'unit': row['unit'],
                    'type': 'technosphere'
                }
                process['exchanges'].append(exchange)

    def populate_biosphere_exchanges(self, biosphere_df, site_id_column='site_id'):
        """
        Populate biosphere exchanges for all activities.
        """
        for key, process in self.lcis.items():
            site_id = process['description'].split('Site ID is ')[-1].strip('. ')
            site_exchanges = biosphere_df[biosphere_df[site_id_column] == site_id]
            for _, row in site_exchanges.iterrows():
                exchange = {
                    'input': row['biosphere_key'],
                    'amount': row['amount'],
                    'unit': row['unit'],
                    'type': 'biosphere'
                }
                process['exchanges'].append(exchange)

    def write_to_database(self):
        """
        Write all activities and exchanges to the Brightway2 database.
        """
        for key, act_data in self.lcis.items():
            act = self.db.new_activity(
                code=key[1],
                **{k: v for k, v in act_data.items() if k != 'exchanges'}
            )
            act.save()
            for exc in act_data['exchanges']:
                act.new_exchange(**exc).save()
        self.db.process()
        print(f"Database '{self.db_name}' has {len(self.db)} activities.")

    def verify_database(self):
        """
        Print a summary of the database contents.
        """
        for act in self.db:
            print(act.key, act.as_dict())
