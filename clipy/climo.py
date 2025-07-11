from datetime import datetime as dt
from noaa_coops import Station
import pandas as pd
import xarray as xr
import numpy as np
import calendar
import yaml
import os

class Data:
    def __init__(self, stationname, stationid, units='metric', timezone='gmt',
                datum='MHHW', hr_threshold=4, day_threshold=2,
                start_date=None, end_date=None,
                redownload=False, reprocess=False, verbose=True):
        """Data class for downloading, formatting, and saving to file
        historical atmospheric (air temperature, barometric pressure, wind) and
        oceanographic (water temperature, water level) data from NOAA CO-OPS
        coastal tide stations.
        WARNING: This may take a while to initiate depending on the amount of
        data to be retrieved.
        
        Inputs:
            stationname: str, desired name of station. Does not need to be the
                CO-OPS name; it is used only for saving data to file.
            stationid: str, NOAA CO-OPS tide station number from which to
                retrieve data
            units: str, either 'metric' or 'english', indicating the units to
                download data in. Defaults to 'metric'.
            timezone: str, one of either 'gmt' for Greenwich Mean Time, 'lst'
                for local standard time, or 'lst_ldt' for adjusted local
                standard/daylight savings time. Defaults to 'gmt'.
            datum: str, tidal datum for water level data. Options: 'STND',
                'MHHW', 'MHW', 'MTL', 'MSL', 'MLW', 'MLLW', 'NAVD'. Defaults to
                'MHHW'.
            hr_threshold: int, maximum number of hours of data that can be
                missing in a given day in order for that day to be included in
                the historical record. Default is 4.
            day_threshold: int, maximum number of days of data that can be
                missing in a given month in order for that month to be included
                in the historical record. Default is 2.
            start_date: str, "YYYY-MM-DD" from which to start downloading data.
                If omitted, data will be downloaded from either the beginning
                of the time series (if no data have previously been downloaded)
                or from the most recent timestamp in the existing data. 
                Defaults to None.
            end_date: str, "YYYY-MM-DD" indicating the last date from which to
                download data. If omitted, "today" is used. Defaults to None.
            redownload: Bool, if True, historical data will be redownloaded and
                the class instance will be re-initiated. Defaults to False.
                WARNING: This may take a while to run depending on the amount
                of data being retrieved.
            reprocess: Bool, if True, existing data will be reprocessed using
                argument variables passed when loading the data and these new
                settings will be written to file, replacing the previous file.
                If False, the class variables will be taken from the existing
                metadata file. Ignored when redownload=True.
            verbose: Bool, print statuses to screen. Defaults to True.
        """
        
        self.name = stationname
        self.dirname = '_'+self.camel(stationname)
        self.id = stationid
        self.unit_system = units.lower()
        self.tz = timezone.lower()
        self.datum = datum.upper()
        self.hr_threshold = hr_threshold
        self.day_threshold = day_threshold
        self.verbose = verbose
        self.variables = []
        today = self._format_date(pd.to_datetime('today'))
        
        # Check for valid arguments
        self._valid_units(self.unit_system)
        self._valid_tz(self.tz)
        self._valid_datum(self.datum)
        
        # Set out directory as a station subdirectory within the home directory
        HOMEDIR = os.getcwd()
        self.outdir = os.path.join(HOMEDIR, self.dirname)

        # =====================================================================
        # If 'redownload' argument is True OR if the directory station name
        # subdirectory does not exist within 'outdir', then create that
        # subdirectory and download historical data.
        if not os.path.exists(self.outdir) or redownload:
            if not os.path.exists(self.outdir):
                if self.verbose:
                    print('Creating new directory for this station.')
                os.makedirs(self.outdir)
        
            # Download all data and save to file
            self.download_data(start_date=start_date, end_date=end_date)
            outFile = os.path.join(self.outdir,
                                   'observational_data_record.csv.gz')
            self.data.to_csv(outFile, compression='infer')
            if self.verbose:
                print(f"Observational data written to file '{outFile}'.")

            # Store units
            deg = u'\N{DEGREE SIGN}'
            self.unit_options = dict({
                'Air Temperature': {'metric': deg+'C', 'english': deg+'F'},
                'Barometric Pressure': {'metric': 'mb', 'english': 'mb'},
                'Wind Speed': {'metric': 'm/s', 'english': 'kn'},
                'Wind Gust': {'metric': 'm/s', 'english': 'kn'},
                'Wind Direction': {'metric': 'deg', 'english': 'deg'},
                'Water Temperature': {'metric': deg+'C', 'english': deg+'F'},
                'Water Level': {'metric': 'm', 'english': 'ft'}
            })
            self.units = {k:v[self.unit_system] \
                          for k, v in self.unit_options.items() \
                          if k in self.variables}
            
            # Save class variables
            self.meta = dict({
                'stationname': self.name,
                'stationid': self.id,
                'dirname': self.dirname,
                'unit_system': self.unit_system,
                'tz': self.tz,
                'datum': self.datum,
                'hr_threshold': self.hr_threshold,
                'day_threshold': self.day_threshold,
                'variables': self.variables,
                'units': self.units,
                'last_obs': {i:self.data[i].last_valid_index().strftime('%Y-%m-%d %X') for i in self.variables},
                'yesterday': {
                    'average': self.daily_avgs().loc[(dt.today()-pd.Timedelta(days=1)).strftime('%Y-%m-%d')].drop('YearDay').to_dict(),
                    'high': self.daily_highs().loc[(dt.today()-pd.Timedelta(days=1)).strftime('%Y-%m-%d')].drop('YearDay').to_dict(),
                    'low': self.daily_lows().loc[(dt.today()-pd.Timedelta(days=1)).strftime('%Y-%m-%d')].drop('YearDay').to_dict()
                    }
                })
            with open(os.path.join(self.outdir, 'metadata.yml'), 'w') as fp:
                yaml.dump(self.meta, fp) 
                    
            # Create and save statistics dictionaries
            self.filtered_hours = \
                pd.concat([self._filter_hours(
                                self.data[var],
                                hr_threshold=self.meta['hr_threshold'])
                           for var in self.variables], axis=1)
            self.filtered_hours = self._DOY(self.filtered_hours)
            self.filtered_days = \
                pd.concat([self._filter_days(
                                self.data[var],
                                hr_threshold=self.meta['hr_threshold'],
                                day_threshold=self.meta['day_threshold'])
                           for var in self.variables], axis=1)
            self.filtered_days = self._DOY(self.filtered_days)
            # Daily stats
            self.daily_records = self.daily_stats()
            statsOutFile = os.path.join(self.outdir, 'statistics-daily.nc')
            self.daily_records.to_netcdf(statsOutFile, mode='w')
            if self.verbose:
                print(f"Observational daily statistics written to '{statsOutFile}'")
            # Monthly stats
            self.monthly_records = self.monthly_stats()
            statsOutFile = os.path.join(self.outdir, 'statistics-monthly.nc')
            self.monthly_records.to_netcdf(statsOutFile, mode='w')
            if self.verbose:
                print(f"Observational monthly statistics written to '{statsOutFile}'")

        # =====================================================================
        # If historical data for this station already exists:
        else:
            # Load the existing metadata from file
            if self.verbose:
                print('Loading metadata from file')
            metaOutFile = os.path.join(self.outdir, 'metadata.yml')
            with open(metaOutFile) as m:
                self.meta = yaml.safe_load(m)
            self._load_from_yaml(self.meta)
            # If data are to be reprocessed using different filtering conditions
            if reprocess:
                # Save new class variables
                self.hr_threshold = hr_threshold
                self.day_threshold = day_threshold
                self.meta['hr_threshold'] = self.hr_threshold
                self.meta['day_threshold'] = self.day_threshold
                with open(metaOutFile, 'w') as fp:
                    yaml.dump(self.meta, fp)
                if verbose:
                    print(f"Saving new class arguments to file {metaOutFile}")
                                
            # Load the historical data from file
            if self.verbose:
                print('Loading historical data from file')
            dataInFile = os.path.join(self.outdir,
                                    'observational_data_record.csv.gz')
            dtypeDict = {k: float for k in self.variables}
            dtypeDict['Water Level QC'] = str
            self.data = pd.read_csv(dataInFile, index_col=f'time_{self.tz}',
                                    parse_dates=True, compression='infer',
                                    dtype=dtypeDict)
                
            # Load daily statistics from file
            if self.verbose:
                print('Loading daily statistics from file')
            statsInFile = os.path.join(self.outdir, 'statistics-daily.nc')
            with xr.open_dataset(statsInFile) as dds:
                self.daily_records = dds.load()
            
            # Load monthly statistics from file
            if self.verbose:
                print('Loading monthly statistics from file')
            statsInFile = os.path.join(self.outdir, 'statistics-monthly.nc')
            with xr.open_dataset(statsInFile) as mds:
                self.monthly_records = mds.load()
            
            # Clean and format
            if self.verbose:
                print('Filtering observational data')
            self.filtered_hours = \
                pd.concat([self._filter_hours(
                                self.data[var],
                                hr_threshold=self.meta['hr_threshold'])
                           for var in self.variables], axis=1)
            self.filtered_hours = self._DOY(self.filtered_hours)
            self.filtered_days = \
                pd.concat([self._filter_days(
                                self.data[var],
                                hr_threshold=self.meta['hr_threshold'],
                                day_threshold=self.meta['day_threshold'])
                           for var in self.variables], axis=1)
            self.filtered_days = self._DOY(self.filtered_days)
        if self.verbose:
            print('Done!')

    # =========================================================================
    def download_data(self, start_date=None, end_date=None):
        """Download data from NOAA CO-OPS"""
        if self.verbose:
            print('Downloading historic data')
        
        # NOAA CO-OPS API
        self.station = Station(id=self.id)

        # List of data variables to combine at the end
        datasets = []

        # If no 'end_date' is passed, download through current day
        if end_date is None:
            end = self._format_date(
                            pd.to_datetime('today') + pd.Timedelta(days=1))
        else:
            end = self._format_date(end_date)

        # Air temperature
        if 'Air Temperature' in self.station.data_inventory:
            self.variables.append('Air Temperature')
            if start_date is None:
                start = self._format_date(
                    self.station.data_inventory['Air Temperature']['start_date'])
            else:
                start = self._format_date(start_date)
            self._load_atemp(start_date=start, end_date=end)
            self.air_temp['atemp_flag'] = self.air_temp['atemp_flag'].str\
                                              .split(',', expand=True)\
                                              .astype(int)\
                                              .sum(axis=1)
            self.air_temp.loc[self.air_temp['atemp_flag'] > 0, 'atemp'] = np.nan
            datasets.append(self.air_temp['atemp'])

        # Water temperature
        if 'Water Temperature' in self.station.data_inventory:
            self.variables.append('Water Temperature')
            if start_date is None:
                start = self._format_date(
                    self.station.data_inventory['Water Temperature']['start_date'])
            else:
                start = self._format_date(start_date)
            self._load_water_temp(start_date=start, end_date=end)
            self.water_temp['wtemp_flag'] = self.water_temp['wtemp_flag'].str\
                                                .split(',', expand=True)\
                                                .astype(int)\
                                                .sum(axis=1)
            self.water_temp.loc[self.water_temp['wtemp_flag'] > 0, 'wtemp'] = np.nan
            datasets.append(self.water_temp['wtemp'])

        # Water level (tides)
        if 'Verified 6-Minute Water Level' in self.station.data_inventory:
            self.variables.append('Water Level')
            if start_date is None:
                start = self._format_date(
                    self.station.data_inventory['Verified 6-Minute Water Level']['start_date'])
            else:
                start = self._format_date(start_date)
            self._load_water_level(start_date=start, end_date=end)
            self.water_levels['wlevel_flag'] = \
                self.water_levels['wlevel_flag'].str.split(',', expand=True)\
                                                .astype(int).sum(axis=1)
            self.water_levels.loc[self.water_levels['wlevel_flag'] > 0, 'wlevel'] = np.nan

            # Hourly water heights (historical product)
            if start_date is None:
                if 'Verified Hourly Height Water Level' in self.station.data_inventory:
                    start = self._format_date(
                        self.station.data_inventory['Verified Hourly Height Water Level']['start_date'])
                    end = self._format_date(
                        self.water_levels.index[0] + pd.Timedelta(days=1))
                    self._load_hourly_height(
                        start_date=start, end_date=end)
                    self.hourly_heights['wlevel_flag'] = \
                        self.hourly_heights['wlevel_flag'].str\
                            .split(',', expand=True).astype(int).sum(axis=1)
                    self.hourly_heights.loc[self.hourly_heights['wlevel_flag'] > 0] = np.nan
                    self.water_levels = pd.concat(
                        (self.hourly_heights[['wlevel', 'wlevel_flag', 'wlevel_qc']][:-1], 
                         self.water_levels[['wlevel', 'wlevel_flag', 'wlevel_qc']]), axis=0)
                    self.water_levels = self.water_levels[~self.water_levels.index.duplicated(keep='first')]
            datasets.append(self.water_levels[['wlevel', 'wlevel_qc']])
            
        # # Barometric pressure
        # if 'Barometric Pressure' in self.station.data_inventory:
        #     self.variables.append('Barometric Pressure')
        #     if start_date is None:
        #         start = self._format_date(
        #             self.station.data_inventory['Barometric Pressure']['start_date'])
        #     else:
        #         start = self._format_date(start_date)
        #     self._load_atm_pres(start_date=start, end_date=end)
        #     self.pressure['apres_flag'] = self.pressure['apres_flag'].str\
        #                                       .split(',', expand=True)\
        #                                       .astype(int).sum(axis=1)
        #     self.pressure.loc[self.pressure['apres_flag'] > 0, 'apres'] = np.nan
        #     datasets.append(self.pressure['apres'])

        # # Wind
        # if 'Wind' in self.station.data_inventory:
        #     self.variables.extend(['Wind Speed', 'Wind Gust'])
        #     if start_date is None:
        #         start = self._format_date(
        #             self.station.data_inventory['Wind']['start_date'])
        #     else:
        #         start = self._format_date(start_date)
        #     self._load_wind(start_date=start, end_date=end)
        #     self.wind['windflag'] = self.wind['wind_flag'].str\
        #                                 .split(',', expand=True).astype(int)\
        #                                 .sum(axis=1)
        #     self.wind.loc[self.wind['wind_flag'] > 0, ['windspeed', 'windgust']] = np.nan
        #     datasets.append(self.wind[['windspeed', 'windgust']])

        # Merge into single dataframe
        if self.verbose:
            print('Compiling data')
        self.data = pd.concat(datasets, axis=1)
        self.data.index.name = f'time_{self.tz}'
        self.data.columns = [i for i in self.variables+['Water Level QC']]

    def update_data(self, start_date=None, end_date=None):
        """Download data from NOAA CO-OPS"""
        if self.verbose:
            print('Downloading latest data')

        # NOAA CO-OPS API
        self.station = Station(id=self.id)

        # List of data variables to combine at the end
        datasets = []
        
        # If no 'start_date' is passed, pick up from the last observation time
        if start_date is None:
            start = self._format_date(self.data.index.max())
        else:
            start = self._format_date(start_date)
            
        # If no 'end_date' is passed, download through end of current date
        if end_date is None:
            end = self._format_date(
                            pd.to_datetime('today') + pd.Timedelta(days=1))
        else:
            end = self._format_date(end_date)
        
        # Air temperature
        if 'Air Temperature' in self.variables:
            # If no 'start_date' is passed, pick up from the last non-NA observation time
            if start_date is None:
                start = self._format_date(self.data['Air Temperature'].last_valid_index())
            else:
                start = self._format_date(start_date)
            # Download data
            self._load_atemp(start_date=start, end_date=end)
            self.air_temp['atemp_flag'] = self.air_temp['atemp_flag'].str\
                                              .split(',', expand=True)\
                                              .astype(int)\
                                              .sum(axis=1)
            self.air_temp.loc[self.air_temp['atemp_flag'] > 0, 'atemp'] = np.nan
            datasets.append(self.air_temp['atemp'])

        # Water temperature
        if 'Water Temperature' in self.variables:
            # If no 'start_date' is passed, pick up from the last non-NA observation time
            if start_date is None:
                start = self._format_date(self.data['Water Temperature'].last_valid_index())
            else:
                start = self._format_date(start_date)
            # Download data
            self._load_water_temp(start_date=start, end_date=end)
            self.water_temp['wtemp_flag'] = self.water_temp['wtemp_flag'].str\
                                                .split(',', expand=True)\
                                                .astype(int)\
                                                .sum(axis=1)
            self.water_temp.loc[self.water_temp['wtemp_flag'] > 0, 'wtemp'] = np.nan
            datasets.append(self.water_temp['wtemp'])

        # Water level (tides)
        if 'Water Level' in self.variables:
            # Check for verified data where preliminary data were previously downloaded
            if self.verbose:
                print('Checking for new verified water level tide data')
            p_start = self.data[self.data['Water Level QC'] == 'p'].index.min()
            p_end = self.data[self.data['Water Level QC'] == 'p'].index.max() + pd.Timedelta(days=1)
            self._load_water_level(start_date=self._format_date(p_start),
                                   end_date=self._format_date(p_end))
            self.water_levels.index.name = self.data.index.name
            self.water_levels.columns = ['Water Level', 's', 'wlevel_flag', 'Water Level QC']
            self.water_levels['wlevel_flag'] = \
                self.water_levels['wlevel_flag'].str.split(',', expand=True)\
                                                .astype(int).sum(axis=1)
            self.water_levels.loc[self.water_levels['wlevel_flag'] > 0, 'Water Level'] = np.nan
            data = self.data.copy()
            data.update(self.water_levels)
            self.data = data
            # If no 'start_date' is passed, pick up from the last non-NA observation time
            if start_date is None:
                start = self._format_date(self.data['Water Level'].last_valid_index())
            else:
                start = self._format_date(start_date)
            # Get latest data
            self._load_water_level(start_date=start, end_date=end)
            self.water_levels['wlevel_flag'] = \
                self.water_levels['wlevel_flag'].str.split(',', expand=True)\
                                                .astype(int).sum(axis=1)
            self.water_levels.loc[self.water_levels['wlevel_flag'] > 0, 'wlevel'] = np.nan
            datasets.append(self.water_levels[['wlevel', 'wlevel_qc']])

        # Barometric pressure
        if 'Barometric Pressure' in self.variables:
            # If no 'start_date' is passed, pick up from the last non-NA observation time
            if start_date is None:
                start = self._format_date(self.data['Barometric Pressure'].last_valid_index())
            else:
                start = self._format_date(start_date)
            # Download data
            self._load_atm_pres(start_date=start, end_date=end)
            self.pressure['apres_flag'] = self.pressure['apres_flag'].str\
                                              .split(',', expand=True)\
                                              .astype(int).sum(axis=1)
            self.pressure.loc[self.pressure['apres_flag'] > 0, 'apres'] = np.nan
            datasets.append(self.pressure['apres'])

        # Wind
        if 'Wind Speed' in self.variables:
            # If no 'start_date' is passed, pick up from the last non-NA observation time
            if start_date is None:
                start = self._format_date(self.data['Wind Speed'].last_valid_index())
            else:
                start = self._format_date(start_date)
            # Download data
            self._load_wind(start_date=start, end_date=end)
            self.wind['windflag'] = self.wind['wind_flag'].str\
                                        .split(',', expand=True).astype(int)\
                                        .sum(axis=1)
            self.wind.loc[self.wind['wind_flag'] > 0, ['windspeed', 'windgust']] = np.nan
            datasets.append(self.wind[['windspeed', 'windgust']])

        # Merge into single dataframe
        data = pd.concat(datasets, axis=1)
        if sum(~data.index.isin(self.data.index)) == 0:
            print('No new data available.')
        else:
            data.index.name = f'time_{self.tz}'
            data.columns = [i for i in self.data.columns]
            data = pd.concat([self.data,
                              data[data.index.isin(self.data.index) == False]],
                             axis=0)
            self.data = data
            self.filtered_hours = \
                pd.concat([self._filter_hours(
                                self.data[var],
                                hr_threshold=self.meta['hr_threshold'])
                           for var in self.variables], axis=1)
            self.filtered_hours = self._DOY(self.filtered_hours)
            self.filtered_days = \
                pd.concat([self._filter_days(
                                self.data[var],
                                hr_threshold=self.meta['hr_threshold'],
                                day_threshold=self.meta['day_threshold'])
                           for var in self.variables], axis=1)
            self.filtered_days = self._DOY(self.filtered_days)
            statsOutFile = os.path.join(self.outdir,
                                        'observational_data_record.csv.gz')
            self.data.to_csv(statsOutFile, compression='infer')
            self.meta['last_obs'] = {i:self.data[i].last_valid_index().strftime('%Y-%m-%d %X') \
                        for i in self.variables}
            if self.verbose:
                print(f"Updated observational data written to file '{statsOutFile}'.")
                print("Done! Run Data.update_stats() to update statistics.")
        
        # Update yesterday's data
        self.meta['yesterday'] = {
            'average': self.daily_avgs().loc[(dt.today()-pd.Timedelta(days=1)).strftime('%Y-%m-%d')].drop('YearDay').to_dict(),
            'high': self.daily_highs().loc[(dt.today()-pd.Timedelta(days=1)).strftime('%Y-%m-%d')].drop('YearDay').to_dict(),
            'low': self.daily_lows().loc[(dt.today()-pd.Timedelta(days=1)).strftime('%Y-%m-%d')].drop('YearDay').to_dict()
            }
        with open(os.path.join(self.outdir, 'metadata.yml'), 'w') as fp:
            yaml.dump(self.meta, fp)
    
    def update_stats(self):    
        """Calculate new statistics and update if any changes"""
        # Daily stats
        _new_daily_stats = self.daily_stats()
        if _new_daily_stats.equals(self.daily_records):
            if self.verbose:
                print('No new daily records set.')
        else:
            if self.verbose:
                print("""Daily stats differ. Updating and saving to file. If new records have been set,
                they will be printed below.\n""")
                try:
                    self._compare(old=self.daily_records, new=_new_daily_stats)
                except ValueError:
                    print("""Cannot display new records. Most likely the records have changed but the years
                    have not, a condition that is not yet supported for printout.""")
                    pass
            self.daily_records = _new_daily_stats
            # Write to file
            statsOutFile = os.path.join(self.outdir, 'statistics-daily.nc')
            self.daily_records.to_netcdf(statsOutFile, mode='w')
            if self.verbose:
                print(f"\nUpdated daily observational statistics written to '{statsOutFile}\n'")
            print('*'*10)

        # Monthly stats
        _new_monthly_stats = self.monthly_stats()
        if _new_monthly_stats.equals(self.monthly_records):
            if self.verbose:
                print('No new monthly records set.')
        else:
            if self.verbose:
                print("""Monthly stats dicts differ. Updating and saving to file. If new records have
                been set, they will be printed below.\n""")
                try:
                    self._compare(old=self.monthly_records,
                                new=_new_monthly_stats)
                except ValueError:
                    print("""Cannot display new records. Most likely the records have changed but the years
                    have not, a condition that is not yet supported for printout.""")
                    pass                
            self.monthly_records = _new_monthly_stats
            # Write to file
            statsOutFile = os.path.join(self.outdir, 'statistics-monthly.nc')
            self.monthly_records.to_netcdf(statsOutFile, mode='w')
            if self.verbose:
                print(f"Updated monthly observational statistics written to '{statsOutFile}'")

    def _format_date(self, datestr):
        """Format date strings into YYYMMDD format"""
        dtdt = pd.to_datetime(datestr)
        return dt.strftime(dtdt, '%Y%m%d')
    
    def camel(self, text):
        """Convert 'text' to camel case"""
        s = text.replace(',', '').replace('.', '').replace("-", " ").replace("_", " ")
        s = s.split()
        if len(text) == 0:
            return text
        return s[0].lower() + ''.join(i.capitalize() for i in s[1:])

    def _valid_units(self, unit):
        valid = {'metric', 'english'}
        if unit.lower() not in valid:
            raise ValueError("units: units must be one of %r." % valid)
    
    def _valid_tz(self, tz):
        valid = {'gmt', 'lst', 'lst_ldt'}
        if tz.lower() not in valid:
            raise ValueError("timezone: timezone must be one of %r." % valid)

    def _valid_datum(self, datum):
        valid = {'STND', 'MHHW', 'MHW', 'MTL', 'MSL', 'MLW', 'MLLW', 'NAVD'}
        if datum.upper() not in valid:
            raise ValueError("datum: datum must be one of %r." % valid)

    def _load_from_yaml(self, blob):
        for k, v in blob.items():
            setattr(self, k, v)
    
    def get_data(self):
        return self.data
        
    def _load_atemp(self, start_date, end_date):
        """Download air temperature data from NOAA CO-OPS from 'start_date'
        through 'end_date'.
        """
        if self.verbose:
            print('Retrieving air temperature data')
        self.air_temp = self.station.get_data(
            begin_date=start_date,
            end_date=end_date,
            product='air_temperature',
            units=self.unit_system,
            time_zone=self.tz)
        self.air_temp.columns = ['atemp', 'atemp_flag']
    
    def _load_wind(self, start_date, end_date):
        """Download wind data from NOAA CO-OPS from 'start_date' through
        'end_date'.
        """
        if self.verbose:
            print('Retrieving wind data')
        self.wind = self.station.get_data(
            begin_date=start_date,
            end_date=end_date,
            product='wind',
            units=self.unit_system,
            time_zone=self.tz)
        self.wind.columns = ['windspeed', 'winddir_deg', 'winddir',
                             'windgust', 'wind_flag']
    
    def _load_atm_pres(self, start_date, end_date):
        """Download barometric pressure data from NOAA CO-OPS from 'start_date'
        through 'end_date'.
        """
        if self.verbose:
            print('Retrieving barometric pressure data')
        self.pressure = self.station.get_data(
            begin_date=start_date,
            end_date=end_date,
            product='air_pressure',
            units=self.unit_system,
            time_zone=self.tz)
        self.pressure.columns = ['apres', 'apres_flag']
    
    def _load_water_temp(self, start_date, end_date):
        """Download water temperature data from NOAA CO-OPS from 'start_date'
       through 'end_date'.
        """
        if self.verbose:
            print('Retrieving water temperature data')
        self.water_temp = self.station.get_data(
            begin_date=start_date,
            end_date=end_date,
            product='water_temperature',
            units=self.unit_system,
            time_zone=self.tz)
        self.water_temp.columns = ['wtemp', 'wtemp_flag']

    def _load_water_level(self, start_date, end_date):
        """Download water level tide data from NOAA CO-OPS from 'start_date'
        through 'end_date'.
        """
        if self.verbose:
            print('Retrieving water level tide data')
        self.water_levels = self.station.get_data(
            begin_date=start_date,
            end_date=end_date,
            product='water_level',
            datum=self.datum,
            units=self.unit_system,
            time_zone=self.tz)
        self.water_levels.columns = ['wlevel', 's', 'wlevel_flag', 'wlevel_qc']

    def _load_hourly_height(self, start_date, end_date):
        """Download verified hourly height data, the predecessor to the water level product, from NOAA CO-OPS from 'start_date' through 'end_date'."""
        if self.verbose:
            print('Retrieving hourly height data')
        self.hourly_heights = self.station.get_data(
            begin_date=start_date,
            end_date=end_date,
            product='hourly_height',
            datum=self.datum,
            units=self.unit_system,
            time_zone=self.tz)
        self.hourly_heights.columns = ['wlevel', 's', 'wlevel_flag']
        # Add QC column for comparing to water level product
        self.hourly_heights['wlevel_qc'] = 'v'
        
    def _DOY(self, df):
        """Calculate year day out of 366"""
        # Day of year as integer
        df['YearDay'] = df.index.day_of_year.astype(int)
        # Years that are NOT leap years
        leapInd = [not calendar.isleap(i) for i in df.index.year]
        # mask = (leapInd) & (df['Month'] > 2)
        mask = (leapInd) & (df.index.month > 2)
        # Advance by one day everything after February 28 
        df.loc[mask, 'YearDay'] += 1
        return df

    def _count_missing_hours(self, group, threshold=3):
        """Return True if the number of hours in a day with good data is 
        greater than or equal to 24-'threshold' (i.e., a 'good' day) and False 
        otherwise.
        """
        num_obs = (~group.resample('1h').mean().isna()).sum()
        good_threshold = 24 - threshold
        return num_obs >= good_threshold

    def _count_missing_days(self, group, threshold=2):
        """Return True if the number of days in a month with good data 
        is greater than or equal to the number of days in the month minus 'theshold' (i.e., a 'good' month) and False
        otherwise.
        """
        try:
            days_in_month = pd.Period(group.index[0].strftime(format='%Y-%m-%d')).days_in_month
            good_days = (~group.resample('1D').mean().isna()).sum()
            good_threshold = days_in_month - threshold
            missing_days_flag = good_days > good_threshold
            return good_days >= good_threshold
        except IndexError:
            pass

    def _filter_hours(self, data, hr_threshold=3):
        """Filter data to remove days with more than 'hr_threshold' missing
        hours of data.
        """
        # Filter out fillVals==31.8
        filtered = data.replace(31.8, np.nan)
        # Filter out days missing more than <hr_threshold> hours
        filtered = filtered.groupby(pd.Grouper(freq='1D')).filter(
            lambda x: self._count_missing_hours(group=x, threshold=hr_threshold))
        return filtered

    def _filter_days(self, data, hr_threshold=3, day_threshold=2):
        """Filter months with more than 'day_threshold' days of missing
        data by first filtering data to remove days with more than 
        'hr_threshold' missing hours of data.
        """
        # Filter out fillVals==31.8
        filtered = data.replace(31.8, np.nan)
        # Filter out days missing more than <hr_threshold> hours
        filtered = self._filter_hours(filtered, hr_threshold=hr_threshold)
        # Filter out months missing more than <day_threshold> days
        filtered = filtered.groupby(pd.Grouper(freq='1M')).filter(
            lambda x: self._count_missing_days(group=x, threshold=day_threshold))
        return filtered

    def daily_highs(self):
        """Daily highs"""
        return self.filtered_hours.groupby(
            pd.Grouper(freq='1D', closed='left', label='left', dropna=True))\
              .max(numeric_only=True)
    
    def daily_lows(self):
        """Daily lows"""
        return self.filtered_hours.groupby(
            pd.Grouper(freq='1D', closed='left', label='left', dropna=True))\
              .min(numeric_only=True)

    def daily_avgs(self, true_average=False):
        """Daily averages by calendar day. If 'true_average' is True, all
        measurements from each 24-hour day will be used to calculate the
        average. Otherwise, only the maximum and minimum observations are used.
        Defaults to False (meteorological standard).
        """
        if true_average:
            return self.filtered_hours.groupby(
                pd.Grouper(freq='1D', closed='left', label='left', dropna=True))\
                  .mean(numeric_only=True)
        else:
            dailyHighs = self.daily_highs()
            dailyLows = self.daily_lows()
            results = (dailyHighs + dailyLows) / 2
            return results

    def mon_daily_highs(self):
        """Daily highs using data filtered by days"""
        return self.filtered_days.groupby(
            pd.Grouper(freq='1D', closed='left', label='left', dropna=True))\
              .max(numeric_only=True)
    
    def mon_daily_lows(self):
        """Daily lows using data filtered by days"""
        return self.filtered_days.groupby(
            pd.Grouper(freq='1D', closed='left', label='left', dropna=True))\
              .min(numeric_only=True)

    def mon_daily_avgs(self, true_average=False):
        """Daily averages by calendar day using data filtered by day. If
        'true_average' is True, all measurements from each 24-hour day will be
        used to calculate the average. Otherwise, only the maximum and minimum
        observations are used. Defaults to False (meteorological standard).
        """
        if true_average:
            return self.filtered_days.groupby(
                pd.Grouper(freq='1D', closed='left', label='left', dropna=True))\
                  .mean(numeric_only=True)
        else:
            dailyHighs = self.mon_daily_highs()
            dailyLows = self.mon_daily_lows()
            results = (dailyHighs + dailyLows) / 2
            return results

    def daily_avg(self, true_average=False):
        """Daily averages. If 'true_average' is True, all measurements from
        each 24-hour day will be used to calculate the daily average.
        Otherwise, only the maximum and minimum observations are used. Defaults
        to False (meteorological standard).
        """
        dailyAvgs = self.daily_avgs(true_average=true_average)
        dailyAvg = dailyAvgs.groupby('YearDay')\
                            .mean(numeric_only=True)
        dailyAvg.index = dailyAvg.index.astype(int)
        results = xr.DataArray(dailyAvg, dims=['yearday', 'variable'])
        results.name = 'Daily Average'
        return results

    def monthly_highs(self, true_average=False):
        """Monthly highs. If 'true_average' is True, all measurements from each
        24-hour day will be used to calculate the daily average. Otherwise,
        only the maximum and minimum observations are used. Defaults to False
        (meteorological standard).
        """
        dailyAvgs = self.mon_daily_avgs(true_average=true_average)
        monthHighs = dailyAvgs.groupby(pd.Grouper(freq='M'))\
                              .max(numeric_only=True)
        return monthHighs
    
    def monthly_lows(self, true_average=False):
        """Monthly lows. If 'true_average' is True, all measurements from each
        24-hour day will be used to calculate the daily average. Otherwise,
        only the maximum and minimum observations are used. Defaults to False
        (meteorological standard).
        """
        dailyAvgs = self.mon_daily_avgs(true_average=true_average)
        monthLows = dailyAvgs.groupby(pd.Grouper(freq='M'))\
                             .min(numeric_only=True)
        return monthLows
    
    def monthly_avg(self, true_average=False):
        """Monthly averages. If 'true_average' is True, all measurements from
        each 24-hour day will be used to calculate the daily average.
        Otherwise, only the maximum and minimum observations are used. Defaults
        to False (meteorological standard).
        """
        dailyAvgs = self.mon_daily_avgs(true_average=true_average)
        monthlyMeans = dailyAvgs.groupby(pd.Grouper(freq='M'))\
                                .mean(numeric_only=True)
        monthlyMeans.drop('YearDay', axis=1, inplace=True)
        monthlyAvg = monthlyMeans.groupby(monthlyMeans.index.month)\
                                 .mean(numeric_only=True)
        monthlyAvg.index = monthlyAvg.index.astype(int)
        results = xr.DataArray(monthlyAvg, dims=['month', 'variable'])
        results.name = 'Monthly Average'
        return results

    def record_high_daily_avg(self, true_average=False):
        """Record high daily averages. If 'true_average' is True, all
        measurements from each 24-hour day will be used to calculate the daily
        average. Otherwise, only the maximum and minimum observations are used.
        Defaults to False (meteorological standard).
        """
        # Calculate the records
        dailyAvgs = self.daily_avgs(true_average=true_average)
        recordHighDailyAvg = \
            dailyAvgs.groupby('YearDay').max(numeric_only=True)
        recordHighDailyAvg.index = recordHighDailyAvg.index.astype(int)
        # Record years
        recordHighDailyAvgYear = dailyAvgs.groupby('YearDay').apply(
            lambda x: x.sort_index().idxmax(numeric_only=True).dt.year)
        recordHighDailyAvgYear.drop('YearDay', axis=1, inplace=True)
        recordHighDailyAvgYear.index = recordHighDailyAvgYear.index.astype(int)
        recordHighDailyAvgYear.columns = \
            [i+' Year' for i in recordHighDailyAvgYear.columns]
        # Create xarray
        results = pd.concat((recordHighDailyAvg, recordHighDailyAvgYear), 
                            axis=1)
        results = xr.DataArray(results, dims=['yearday', 'variable'])
        results.name = 'Record High Daily Average'
        return results

    def record_high_monthly_avg(self, true_average=False):
        """Record high monthly averages. If 'true_average' is True, all
        measurements from each 24-hour day will be used to calculate the daily
        average. Otherwise, only the maximum and minimum observations are used.
        Defaults to False (meteorological standard).
        """
        # Calculate the records
        dailyAvgs = self.mon_daily_avgs(true_average=true_average)
        monthlyAvgs = dailyAvgs.groupby(pd.Grouper(freq='M'))\
                               .mean(numeric_only=True)
        monthlyAvgs.drop('YearDay', axis=1, inplace=True)
        recordHighMonthlyAvg = monthlyAvgs.groupby(monthlyAvgs.index.month)\
                                          .max(numeric_only=True)
        recordHighMonthlyAvg.index = recordHighMonthlyAvg.index.astype(int)
        # Record years
        recordHighMonthlyAvgYear = \
            monthlyAvgs.groupby(monthlyAvgs.index.month).apply(
                lambda x: x.sort_index().idxmax(numeric_only=True).dt.year)
        recordHighMonthlyAvgYear.index = \
            recordHighMonthlyAvgYear.index.astype(int)
        recordHighMonthlyAvgYear.columns = \
            [i+' Year' for i in recordHighMonthlyAvgYear.columns]
        # Create xarray
        results = pd.concat((recordHighMonthlyAvg, recordHighMonthlyAvgYear),
                             axis=1)
        results = xr.DataArray(results, dims=['month', 'variable'])
        results.name = 'Record High Monthly Average'
        return results

    def record_low_daily_avg(self, true_average=False):
        """Record low daily averages.  If 'true_average' is True, all
        measurements from each 24-hour day will be used to calculate the
        average. Otherwise, only the maximum and minimum observations are used.
        Defaults to False (meteorological standard)."""
        # Calculate the records
        dailyAvgs = self.daily_avgs(true_average=true_average)
        recordLowDailyAvg = \
            dailyAvgs.groupby('YearDay').min(numeric_only=True)
        recordLowDailyAvg.index = recordLowDailyAvg.index.astype(int)
        # Record years
        recordLowDailyAvgYear = dailyAvgs.groupby('YearDay').apply(
            lambda x: x.sort_index().idxmin(numeric_only=True).dt.year)
        recordLowDailyAvgYear.drop('YearDay', axis=1, inplace=True)
        recordLowDailyAvgYear.index = recordLowDailyAvgYear.index.astype(int)
        recordLowDailyAvgYear.columns = \
            [i+' Year' for i in recordLowDailyAvgYear.columns]
        # Create xarray
        results = pd.concat((recordLowDailyAvg, recordLowDailyAvgYear), axis=1)
        results = xr.DataArray(results, dims=['yearday', 'variable'])
        results.name = 'Record Low Daily Average'
        return results

    def record_low_monthly_avg(self, true_average=False):
        """Record low monthly averages. If 'true_average' is True, all
        measurements from each 24-hour day will be used to calculate the daily
        average. Otherwise, only the maximum and minimum observations are used.
        Defaults to False (meteorological standard).
        """
        # Calculate the records
        dailyAvgs = self.mon_daily_avgs(true_average=true_average)
        monthlyAvgs = dailyAvgs.groupby(pd.Grouper(freq='M'))\
                               .mean(numeric_only=True)
        monthlyAvgs.drop('YearDay', axis=1, inplace=True)
        recordLowMonthlyAvg = \
            monthlyAvgs.groupby(monthlyAvgs.index.month).min(numeric_only=True)
        recordLowMonthlyAvg.index = recordLowMonthlyAvg.index.astype(int)
        # Record years
        recordLowMonthlyAvgYear = \
            monthlyAvgs.groupby(monthlyAvgs.index.month).apply(
                lambda x: x.sort_index().idxmin(numeric_only=True).dt.year)
        recordLowMonthlyAvgYear.index = \
            recordLowMonthlyAvgYear.index.astype(int)
        recordLowMonthlyAvgYear.columns = \
            [i+' Year' for i in recordLowMonthlyAvgYear.columns]
        # Create xarray
        results = pd.concat((recordLowMonthlyAvg, recordLowMonthlyAvgYear),
                             axis=1)
        results = xr.DataArray(results, dims=['month', 'variable'])
        results.name = 'Record Low Monthly Average'
        return results

    def avg_daily_high(self):
        """Average daily highs."""        
        dailyHighs = self.daily_highs()
        results = dailyHighs.groupby('YearDay')\
                            .mean(numeric_only=True)
        results = xr.DataArray(results, dims=['yearday', 'variable'])
        results.name = 'Average Daily High'
        return results

    def avg_monthly_high(self, true_average=False):
        """Average monthly highs. If 'true_average' is True, all measurements
        from each 24-hour day will be used to calculate the daily average.
        Otherwise, only the maximum and minimum observations are used. Defaults
        to False (meteorological standard).
        """
        monthlyHighs = self.monthly_highs(true_average=true_average)
        monthlyHighs.drop('YearDay', axis=1, inplace=True)
        avgMonthlyHighs = monthlyHighs.groupby(monthlyHighs.index.month)\
                                      .mean(numeric_only=True)
        results = xr.DataArray(avgMonthlyHighs, dims=['month', 'variable'])
        results.name = 'Average Monthly High'
        return results

    def lowest_daily_high(self):
        """Lowest daily highs."""
        # Calculate the record
        dailyHighs = self.daily_highs()
        lowestHigh = dailyHighs.groupby('YearDay')\
                               .min(numeric_only=True)
        lowestHigh.index = lowestHigh.index.astype(int)
        # Record years
        lowestHighYear = dailyHighs.groupby('YearDay').apply(
            lambda x: x.sort_index().idxmin(numeric_only=True).dt.year)
        lowestHighYear.drop('YearDay', axis=1, inplace=True)
        lowestHighYear.index = lowestHighYear.index.astype(int)
        lowestHighYear.columns = \
            [i+' Year' for i in lowestHighYear.columns]
        # Create xarray
        results = pd.concat((lowestHigh, lowestHighYear), axis=1)
        results = xr.DataArray(results, dims=['yearday', 'variable'])
        results.name = 'Lowest Daily High'
        return results

    def lowest_monthly_high(self, true_average=False):
        """Lowest monthly highs. If 'true_average' is True, all measurements
        from each 24-hour day will be used to calculate the daily average.
        Otherwise, only the maximum and minimum observations are used. Defaults
        to False (meteorological standard).
        """
        # Calculate the record
        monthlyHighs = self.monthly_highs(true_average=true_average)
        monthlyHighs.drop('YearDay', axis=1, inplace=True)
        lowMonthlyHigh = monthlyHighs.groupby(monthlyHighs.index.month)\
                                     .min(numeric_only=True)
        lowMonthlyHigh.index = lowMonthlyHigh.index.astype(int)
        # Record years
        lowMonthlyHighYear = \
            monthlyHighs.groupby(monthlyHighs.index.month).apply(
                lambda x: x.sort_index().idxmin(numeric_only=True).dt.year)
        lowMonthlyHighYear.index = lowMonthlyHighYear.index.astype(int)
        lowMonthlyHighYear.columns = \
            [i+' Year' for i in lowMonthlyHighYear.columns]
        # Create xarray
        results = pd.concat((lowMonthlyHigh, lowMonthlyHighYear), axis=1)
        results = xr.DataArray(results, dims=['month', 'variable'])
        results.name = 'Lowest Monthly High'
        return results

    def record_daily_high(self):
        """Record daily highs."""
        # Calculate the record
        dailyHighs = self.daily_highs()
        recordHigh = dailyHighs.groupby('YearDay')\
                               .max(numeric_only=True)
        recordHigh.index = recordHigh.index.astype(int)
        # Record years
        recordHighYear = dailyHighs.groupby('YearDay').apply(
            lambda x: x.sort_index().idxmax(numeric_only=True).dt.year)
        recordHighYear.drop('YearDay', axis=1, inplace=True)
        recordHighYear.index = recordHighYear.index.astype(int)
        recordHighYear.columns = \
            [i+' Year' for i in recordHighYear.columns]
        # Create xarray
        results = pd.concat((recordHigh, recordHighYear), axis=1)
        results = xr.DataArray(results, dims=['yearday', 'variable'])
        results.name = 'Record Daily High'
        return results

    def record_monthly_high(self, true_average=False):
        """Record monthly highs. If 'true_average' is True, all measurements
        from each 24-hour day will be used to calculate the daily average.
        Otherwise, only the maximum and minimum observations are used. Defaults
        to False (meteorological standard).
        """
        # Calculate the record
        monthlyHighs = self.monthly_highs(true_average=true_average)
        monthlyHighs.drop('YearDay', axis=1, inplace=True)
        recordMonthlyHigh = monthlyHighs.groupby(monthlyHighs.index.month)\
                                        .max(numeric_only=True)
        recordMonthlyHigh.index = recordMonthlyHigh.index.astype(int)
        # Record years
        recordMonthlyHighYear = \
            monthlyHighs.groupby(monthlyHighs.index.month).apply(
                lambda x: x.sort_index().idxmax(numeric_only=True).dt.year)
        recordMonthlyHighYear.index = recordMonthlyHighYear.index.astype(int)
        recordMonthlyHighYear.columns = \
            [i+' Year' for i in recordMonthlyHighYear.columns]
        # Create xarray
        results = pd.concat((recordMonthlyHigh, recordMonthlyHighYear), axis=1)
        results = xr.DataArray(results, dims=['month', 'variable'])
        results.name = 'Record Monthly High'
        return results

    def avg_daily_low(self):
        """Average daily lows."""        
        dailyLows = self.daily_lows()
        results = dailyLows.groupby('YearDay')\
                           .mean(numeric_only=True)
        results = xr.DataArray(results, dims=['yearday', 'variable'])
        results.name = 'Average Daily Low'
        return results

    def avg_monthly_low(self, true_average=False):
        """Average monthly lows. If 'true_average' is True, all measurements
        from each 24-hour day will be used to calculate the daily average.
        Otherwise, only the maximum and minimum observations are used. Defaults
        to False (meteorological standard).
        """
        monthlyLows = self.monthly_lows(true_average=true_average)
        monthlyLows.drop('YearDay', axis=1, inplace=True)
        avgMonthlyLows = monthlyLows.groupby(monthlyLows.index.month)\
                                    .mean(numeric_only=True)
        results = xr.DataArray(avgMonthlyLows, dims=['month', 'variable'])
        results.name = 'Average Monthly Low'
        return results

    def highest_daily_low(self):
        """Highest daily lows."""
        # Calculate the record
        dailyLows = self.daily_lows()
        highestLow = dailyLows.groupby('YearDay')\
                              .max(numeric_only=True)
        highestLow.index = highestLow.index.astype(int)
        # Record years
        highestLowYear = dailyLows.groupby('YearDay').apply(
            lambda x: x.sort_index().idxmax(numeric_only=True).dt.year)
        highestLowYear.drop('YearDay', axis=1, inplace=True)
        highestLowYear.index = highestLowYear.index.astype(int)
        highestLowYear.columns = [i+' Year' for i in highestLowYear.columns]
        # Create xarray
        results = pd.concat((highestLow, highestLowYear), axis=1)
        results = xr.DataArray(results, dims=['yearday', 'variable'])
        results.name = 'Highest Daily Low'
        return results

    def highest_monthly_low(self, true_average=False):
        """Highest monthly lows. If 'true_average' is True, all measurements
        from each 24-hour day will be used to calculate the daily average.
        Otherwise, only the maximum and minimum observations are used. Defaults
        to False (meteorological standard).
        """
        # Calculate the record
        monthlyLows = self.monthly_lows(true_average=true_average)
        monthlyLows.drop('YearDay', axis=1, inplace=True)
        highestMonthlyLow = monthlyLows.groupby(monthlyLows.index.month)\
                                       .max(numeric_only=True)
        highestMonthlyLow.index = highestMonthlyLow.index.astype(int)
        # Record years
        highestMonthlyLowYear = \
            monthlyLows.groupby(monthlyLows.index.month).apply(
                lambda x: x.sort_index().idxmax(numeric_only=True).dt.year)
        highestMonthlyLowYear.index = highestMonthlyLowYear.index.astype(int)
        highestMonthlyLowYear.columns = \
            [i+' Year' for i in highestMonthlyLowYear.columns]
        # Create xarray
        results = pd.concat((highestMonthlyLow, highestMonthlyLowYear), axis=1)
        results = xr.DataArray(results, dims=['month', 'variable'])
        results.name = 'Highest Monthly Low'
        return results

    def record_daily_low(self):
        """Record daily lows."""
        # Calculate the record
        dailyLows = self.daily_lows()
        recordLow = dailyLows.groupby('YearDay')\
                             .min(numeric_only=True)
        recordLow.index = recordLow.index.astype(int)
        # Record years
        recordLowYear = dailyLows.groupby('YearDay').apply(
            lambda x: x.sort_index().idxmin(numeric_only=True).dt.year)
        recordLowYear.drop('YearDay', axis=1, inplace=True)
        recordLowYear.index = recordLowYear.index.astype(int)
        recordLowYear.columns = [i+' Year' for i in recordLowYear.columns]
        # Create xarray
        results = pd.concat((recordLow, recordLowYear), axis=1)
        results = xr.DataArray(results, dims=['yearday', 'variable'])
        results.name = 'Record Daily Low'
        return results

    def record_monthly_low(self, true_average=False):
        """Record monthly lows. If 'true_average' is True, all measurements
        from each 24-hour day will be used to calculate the daily average.
        Otherwise, only the maximum and minimum observations are used. Defaults
        to False (meteorological standard).
        """
        # Calculate the record
        monthlyLows = self.monthly_lows(true_average=true_average)
        monthlyLows.drop('YearDay', axis=1, inplace=True)
        recordMonthlyLow = monthlyLows.groupby(monthlyLows.index.month)\
                                      .min(numeric_only=True)
        recordMonthlyLow.index = recordMonthlyLow.index.astype(int)
        # Record years
        recordMonthlyLowYear = \
            monthlyLows.groupby(monthlyLows.index.month).apply(
                lambda x: x.sort_index().idxmin(numeric_only=True).dt.year)
        recordMonthlyLowYear.index = recordMonthlyLowYear.index.astype(int)
        recordMonthlyLowYear.columns = \
            [i+' Year' for i in recordMonthlyLowYear.columns]
        # Create xarray
        results = pd.concat((recordMonthlyLow, recordMonthlyLowYear), axis=1)
        results = xr.DataArray(results, dims=['month', 'variable'])
        results.name = 'Record Monthly Low'
        return results

    def number_of_years_byday(self):
        """Number of years in the historical data records by day of year."""
        numYears = pd.concat(
            [self.filtered_hours[[v, 'YearDay']]\
                .dropna().groupby('YearDay').apply(
                    lambda x: len(x.index.year.unique())) \
             for v in self.filtered_hours.columns if v != 'YearDay'], axis=1)
        numYears.columns = [v for v in self.filtered_hours.columns \
                            if v != 'YearDay']
        results = xr.DataArray(numYears, dims=['yearday', 'variable'])
        results.name = 'Number of Years'
        return results

    def number_of_years_bymonth(self):
        """Number of years in the historical data records by month."""
        numYears = pd.concat(
            [self.filtered_days[v]\
                 .dropna().groupby(self.filtered_days[v].dropna().index.month).apply(
                    lambda x: len(x.index.year.unique())) \
                for v in self.filtered_days if v != 'YearDay'], axis=1)
        numYears.columns = [v for v in self.filtered_days.columns \
                            if v != 'YearDay']
        results = xr.DataArray(numYears, dims=['month', 'variable'])
        results.name = 'Number of Years'
        return results
    
    def generate_yeardays(self):
        return pd.date_range(start='2020-01-01', end='2020-12-31', freq='1D')\
                 .strftime('%d-%b')
    
    def daily_stats(self):
        """Create xarray of daily statistics for all science variables"""
        daily_records = xr.Dataset(
            {'Daily Average': self.daily_avg(),
             'Record High Daily Average': self.record_high_daily_avg(),
             'Record Low Daily Average': self.record_low_daily_avg(),
             'Average High': self.avg_daily_high(),
             'Lowest High': self.lowest_daily_high(),
             'Record High': self.record_daily_high(),
             'Average Low': self.avg_daily_low(),
             'Highest Low': self.highest_daily_low(),
             'Record Low': self.record_daily_low(),
             'Years': self.number_of_years_byday()},
            attrs = {k:v for k, v in self.meta.items() \
                     if k not in ['outdir', 'variables', 'units', 'last_obs', 'yesterday']})
        
        # Add data units for each variable to the array as metadata attributes
        for k, v in self.meta['units'].items():
            daily_records.attrs[k+' units'] = v
        
        # Add time series ranges for each variable to the array as metadata 
        # attributes
        for var in daily_records.coords['variable'].values:
            if 'Year' not in var:
                daily_records.attrs[var+' data range'] = \
                    (self.filtered_hours[var].first_valid_index().strftime('%Y-%m-%d'),
                     self.filtered_hours[var].last_valid_index().strftime('%Y-%m-%d'))
        
        # Rearrange array coordinates and variables: separate records and 
        # years into smaller arrays
        day_records = daily_records.sel(variable = [
            i for i in daily_records.coords['variable'].values \
                if 'Year' not in i])
        day_years = daily_records.sel(variable = [
            i for i in daily_records.coords['variable'].values if 'Year' in i])
        
        # Add "Year" to variable names and remove it from coordinate name
        day_years = day_years.rename_vars(
            {i:i+' Year' for i in day_years.data_vars})
        day_years.coords['variable'] = \
            [i.removesuffix(' Year') for i in day_years.coords['variable'].values]

        # Merge arrays together
        daily_records = xr.merge([day_records, day_years])
        daily_records = daily_records[
            [item for items in zip(day_records.data_vars, day_years.data_vars) \
             for item in items]]
        daily_records = daily_records.drop_vars(
            [x for x in daily_records.data_vars \
             if daily_records[x].isnull().all()])
        
        # Convert years to integers
        daily_records[[i for i in daily_records.data_vars if "Year" in i]] = \
            daily_records[
            [i for i in daily_records.data_vars if "Year" in i]].astype(int)

        # Replace yearday with calendar day and rename coordinate
        daily_records.coords['yearday'] = \
            pd.date_range(start='2020-01-01', end='2020-12-31', freq='1D')\
              .strftime('%d-%b')
        daily_records = daily_records.rename({'yearday':'Date'})

        return daily_records

    def monthly_stats(self):
        """Create xarray of monthly statistics for all science variables"""
        monthly_records = xr.Dataset(
            {'Monthly Average': self.monthly_avg(),
             'Record High Monthly Average': self.record_high_monthly_avg(),
             'Record Low Monthly Average': self.record_low_monthly_avg(),
             'Average High': self.avg_monthly_high(),
             'Lowest High': self.lowest_monthly_high(),
             'Record High': self.record_monthly_high(),
             'Average Low': self.avg_monthly_low(),
             'Highest Low': self.highest_monthly_low(),
             'Record Low': self.record_monthly_low(),
             'Years': self.number_of_years_bymonth()},
            attrs = {k:v for k, v in self.meta.items() \
                     if k not in ['outdir', 'variables', 'units', 'last_obs', 'yesterday']})

        # Add data units for each variable to the array as metadata attributes
        for k, v in self.meta['units'].items():
            monthly_records.attrs[k+' units'] = v

        # Add time series ranges for each variable to the array as metadata 
        # attributes
        for var in monthly_records.coords['variable'].values:
            if 'Year' not in var:
                monthly_records.attrs[var+' data range'] = \
                    (self.filtered_days[var].first_valid_index().strftime('%Y-%m-%d'),
                     self.filtered_days[var].last_valid_index().strftime('%Y-%m-%d'))
        
        # Rearrange array coordinates and variables: separate records and 
        # years into smaller arrays
        mon_records = monthly_records.sel(variable = [
            i for i in monthly_records.coords['variable'].values \
                if 'Year' not in i])
        mon_years = monthly_records.sel(variable = [
            i for i in monthly_records.coords['variable'].values \
                if 'Year' in i])
        
        # Add "Year" to variable names and remove it from coordinate name
        mon_years = mon_years.rename_vars(
            {i:i+' Year' for i in mon_years.data_vars})
        mon_years.coords['variable'] = \
            [i.removesuffix(' Year') for i in mon_years.coords['variable'].values]

        # Merge arrays together
        monthly_records = xr.merge([mon_records, mon_years])
        monthly_records = monthly_records[
            [item for items in zip(mon_records.data_vars, mon_years.data_vars) for item in items]]
        monthly_records = monthly_records.drop_vars(
            [x for x in monthly_records.data_vars if monthly_records[x].isnull().all()])
        
        # Convert years to integers
        monthly_records[[i for i in monthly_records.data_vars if "Year" in i]] = \
            monthly_records[[i for i in monthly_records.data_vars if "Year" in i]].astype(int)

        # Replace yearday with calendar day and rename coordinate
        monthly_records.coords['month'] = \
            pd.date_range(start='2020-01-01', end='2020-12-31', freq='1m')\
              .strftime('%b')
        monthly_records = monthly_records.rename({'month': 'Month'})

        return monthly_records

    def _compare(self, old, new):
        """Compare 'old' and 'new' records xarrays excluding daily highs, lows,
        and averages, since these will always change with updated data.
        """
        # Mask out unchanged records
        diffs = old != new
        exclude = ['Daily Average', 'Monthly Average', 'Average High',
                   'Average Low', 'Years']
        try:
            new_records = xr.where(diffs, new, np.nan)\
                            .drop_vars(exclude, errors='ignore')\
                            .dropna(dim='variable', how='all')\
                            .dropna(dim='Date', how='all')\
                            .to_dataframe()
        except ValueError:
            new_records = xr.where(diffs, new, np.nan)\
                            .drop_vars(exclude, errors='ignore')\
                            .dropna(dim='variable', how='all')\
                            .dropna(dim='Month', how='all')\
                            .to_dataframe()
        
        # List of records set
        record_set = new_records.loc[:, new_records.columns.str.endswith('Year')].columns
        record_set = [i.replace(' Year', '') for i in record_set]

        # Loop through the dataframe of new records to report out the updates
        # This is a hacky solution that pieces together the various compoents
        # of the iterrows() output.
        for row in new_records.iterrows():
            for record in record_set:
                if not np.isnan(row[1][record]):
                    var = row[0][1]
                    newRecord = row[1][record]
                    newYear = int(row[1][record+' Year'])
                    try:
                        oldRecord = old.sel(variable=var,
                                            Date=row[0][0])[record].values
                        oldYear = old.sel(variable=var,
                                          Date=row[0][0])[record+' Year'].values
                    except KeyError:
                        oldRecord = old.sel(variable=var,
                                            Month=row[0][0])[record].values
                        oldYear = old.sel(variable=var,
                                          Month=row[0][0])[record+' Year'].values
                    units = self.units[var]
                    print(f"{record.capitalize()} {var.lower()} set {row[0][0]} {newYear}:\n\t"\
                        f"{np.round(newRecord, 3)} {units} (previously {np.round(oldRecord, 3)} {units} in {oldYear})")
    
    def get_daily_stats(self, var=None):
        """Return the daily statistics for variable 'var'"""
        try:
            return self.daily_records.sel(variable=var)
        except AttributeError:
            raise AttributeError(
                """Instance of Data has no daily stats yet. Run Data.stats() to
                calculate stats and try again.""")
    
    def get_monthly_stats(self, var=None):
        """Return the monthly statistics dictionary"""
        try:
            return self.monthly_records.sel(variable=var)
        except AttributeError:
            raise AttributeError(
                """Instance of Data has no monthly stats yet. Run Data.stats() to calculate stats and try again.""")

    def get_daily_stats_table(self, var=None):
        """Return the daily statistics table"""
        try:
            return self.daily_records.sel(variable=var)\
                                     .to_dataframe().drop('variable', axis=1)
        except AttributeError:
            raise AttributeError(
                """Instance of Data has no daily stats table yet. Run Data.daily_stats() to calculate stats and try again.""")
            
    def get_monthly_stats_table(self, var=None):
        """Return the monthly statistics table"""
        try:
            return self.monthly_records.sel(variable=var)\
                                       .to_dataframe().drop('variable', axis=1)
        except AttributeError:
            raise AttributeError(
                """Instance of Data has no monthly stats table yet. Run Data.monthly_stats() to calculate stats and try again.""")

    def _skip_keys(self, d, keys):
        return {x: d[x] for x in d if x not in keys}

    def set_station(self, station):
        self.name = station
        
    def get_station(self):
        return self.name
    
    def get_stationid(self):
        return self.id
    
    def get_variables(self):
        return self.variables
    
    def __str__(self):
        return(
            f"""Oceanic and atmospheric observations for station '{self.name}'   
            (station ID {self.id}): {self.station.data_inventory}"""
            )
    
    def __repr__(self):
        return(
            f"""{type(self).__name__}(stationname='{self.name}', stationid='{self.id}',
            timezone='{self.tz}', units='{self.units}', datum='{self.datum}', hr_threshold='{self.hr_threshold}', day_threshold='{self.day_threshold}')"""
            )