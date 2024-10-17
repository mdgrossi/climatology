from noaa_coops import Station
import datetime as dt
import pandas as pd
import xarray as xr
import numpy as np
import calendar
import yaml
import os

class Data:
    def __init__(self, stationname, stationid, units='metric', timezone='gmt',
                datum='MHHW', outdir=None, hr_threshold=3, day_threshold=2,
                redownload=False, verbose=True):
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
            outdir: str, directory to save data to. Defaults to present working
                directory.
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
                the historical record. Default is 3.
            day_threshold: int, maximum number of days of data that can be
                missing in a given month in order for that month to be included
                in the historical record. Default is 2.
            redownload: Bool, if True, historical data will be redownloaded and
                the class instance will be re-initiated. Defaults to False.
                WARNING: This may take a while to run depending on the amount
                of data being retrieved.
            verbose: Bool, print statuses to screen. Defaults to True.
        """
        
        self.name = stationname
        self.dirname = self.camel(stationname)
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
        
        # Set data directory, creating station subdirectory if needed
        if outdir:
            self.outdir = os.path.join(outdir, self.dirname)
        else:
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
            self.download_data(start_date=None, end_date=None)
            outFile = os.path.join(self.outdir,
                                   'observational_data_record.csv.gz')
            self.data.to_csv(outFile, compression='infer')
            self.data = self._DOY(self.data)
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
                'outdir': self.dirname,
                'unit_system': self.unit_system,
                'tz': self.tz,
                'datum': self.datum,
                'hr_threshold': self.hr_threshold,
                'day_threshold': self.day_threshold,
                'variables': self.variables,
                'units': self.units})
            with open(os.path.join(self.outdir, 'metadata.yml'), 'w') as fp:
                yaml.dump(self.meta, fp) 
                    
            # Create and save statistics dictionaries
            self.filtered_data = \
                pd.concat([self._filter_data(self.data[var],
                                            hr_threshold=self.meta['hr_threshold'],
                                            day_threshold=self.meta['day_threshold'])
                                       for var in self.variables], axis=1)
            self.filtered_data = self._DOY(self.filtered_data)
            # Daily stats
            self.daily_records = self.daily_stats()
            statsOutFile = os.path.join(self.outdir, 'statistics-daily.nc')
            self.daily_records.to_netcdf(statsOutFile, mode='w')
            if self.verbose:
                print("Observational daily statistics written to "\
                      f"'{statsOutFile}'")
            # Monthly stats
            self.monthly_records = self.monthly_stats()
            statsOutFile = os.path.join(self.outdir, 'statistics-monthly.nc')
            self.monthly_records.to_netcdf(statsOutFile, mode='w')
            if self.verbose:
                print("Observational monthly statistics written to "
                      f"'{statsOutFile}'")            

        # =====================================================================
        # If historical data for this station already exists:
        else:
            # Load the metadata from file
            if self.verbose:
                print('Loading metadata from file')
            with open(os.path.join(self.outdir, 'metadata.yml')) as m:
                self.meta = yaml.safe_load(m)
            self._load_from_yaml(self.meta)
            self.outdir = os.path.join(os.getcwd(), self.meta['outdir'])
            
            # Load the historical data from file
            if self.verbose:
                print('Loading historical data from file')
            dataInFile = os.path.join(self.outdir,
                                  'observational_data_record.csv.gz')
            self.data = pd.read_csv(dataInFile, index_col=f'time_{self.tz}',
                                    parse_dates=True, compression='infer')
            self.data = self._DOY(self.data)
            
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
            self.filtered_data = \
                pd.concat([self._filter_data(self.data[var],
                                            hr_threshold=self.meta['hr_threshold'],
                                            day_threshold=self.meta['day_threshold'])
                                       for var in self.variables], axis=1)
            self.filtered_data = self._DOY(self.filtered_data)
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
        if not end_date:
            end_date = self._format_date(pd.to_datetime('today')) + \
                       pd.Timedelta(days=1)

        # Air temperature
        if 'Air Temperature' in self.station.data_inventory:
            self.variables.append('Air Temperature')
            if not start_date:
                start_date = self._format_date(
                    self.station.data_inventory['Air Temperature']['start_date'])
            self._load_atemp(start_date=start_date, end_date=end_date)
            self.air_temp['atemp_flag'] = self.air_temp['atemp_flag'].str\
                                                .split(',', expand=True)\
                                                .astype(int)\
                                                .sum(axis=1)
            self.air_temp.loc[self.air_temp['atemp_flag']>0, 'atemp'] = np.nan
            datasets.append(self.air_temp['atemp'])

        # # Barometric pressure
        # if 'Barometric Pressure' in self.station.data_inventory:
        #     self.variables.append('Barometric Pressure')
        #     if not start_date:
        #         start_date = self._format_date(self.station.data_inventory['Barometric Pressure']['start_date'])
        #     self._load_atm_pres(start_date=start_date, end_date=end_date)
            # self.pressure['apres_flag'] = self.pressure['apres_flag'].str.split(',', expand=True).astype(int).sum(axis=1)
            # self.pressure.loc[self.pressure['apres_flag'] > 0, 'apres'] = np.nan
        #     datasets.append(self.pressure['apres'])

        # # Wind
        # if 'Wind' in self.station.data_inventory:
        #     self.variables.extend(['Wind Speed', 'Wind Gust'])
        #     if not start_date:
        #         start_date = self._format_date(self.station.data_inventory['Wind']['start_date'])
        #     self._load_wind(start_date=start_date, end_date=end_date)
            # self.wind['windflag'] = self.wind['wind_flag'].str.split(',', expand=True).astype(int).sum(axis=1)
            # self.wind.loc[self.wind['wind_flag'] > 0, ['windspeed', 'windgust']] = np.nan
        #     datasets.append(self.wind[['windspeed', 'windgust']])

        # Water temperature
        if 'Water Temperature' in self.station.data_inventory:
            self.variables.append('Water Temperature')
            if not start_date:
                start_date = self._format_date(
                    self.station.data_inventory['Water Temperature']['start_date'])
            self._load_water_temp(start_date=start_date, end_date=end_date)
            self.water_temp['wtemp_flag'] = self.water_temp['wtemp_flag'].str\
                                                    .split(',', expand=True)\
                                                    .astype(int)\
                                                    .sum(axis=1)
            self.water_temp.loc[self.water_temp['wtemp_flag']>0, 'wtemp'] = np.nan
            datasets.append(self.water_temp['wtemp'])

        # # Water level (tides)
        # if 'Verified 6-Minute Water Level' in self.station.data_inventory:
        #     self.variables.append('Water Level')
        #     if not start_date:
        #         start_date = self._format_date(self.station.data_inventory['Verified 6-Minute Water Level']['start_date'])
        #     self._load_water_level(start_date=start_date, end_date=end_date)
            # self.water_levels['wlevel_flag'] = self.water_levels['wlevel_flag'].str.split(',', expand=True).astype(int).sum(axis=1)
            # self.water_levels.loc[self.water_levels['wlevel_flag'] > 0, 'wlevel'] = np.nan
        #     datasets.append(self.water_levels['wlevel'])

        # Merge into single dataframe
        if self.verbose:
            print('Compiling data')
        self.data = pd.concat(datasets, axis=1)
        self.data.index.name = f'time_{self.tz}'
        self.data.columns = [i for i in self.variables]

    def update_data(self, start_date=None, end_date=None):
        """Download data from NOAA CO-OPS"""
        if self.verbose:
            print('Downloading latest data')

        # NOAA CO-OPS API
        self.station = Station(id=self.id)

        # List of data variables to combine at the end
        datasets = []
        
        # If no 'start_date' is passed, pick up from the last observation time
        if not start_date:
            start_date = self._format_date(self.data.index.max())
            
        # If no 'end_date' is passed, download through end of current date
        if not end_date:
            end_date = self._format_date(pd.to_datetime('today') + pd.Timedelta(days=1))
        
        # Air temperature
        if 'Air Temperature' in self.variables:
            self._load_atemp(start_date=start_date, end_date=end_date)
            datasets.append(self.air_temp['atemp'])

        # Barometric pressure
        if 'Barometric Pressure' in self.variables:
            self._load_atm_pres(start_date=start_date, end_date=end_date)
            datasets.append(self.pressure['apres'])

        # Wind
        if 'Wind Speed' in self.variables:
            self._load_wind(start_date=start_date, end_date=end_date)
            datasets.append(self.wind[['windspeed', 'windgust']])

        # Water temperature
        if 'Water Temperature' in self.variables:
            self._load_water_temp(start_date=start_date, end_date=end_date)
            datasets.append(self.water_temp['wtemp'])

        # Water level (tides)
        if 'Verified 6-Minute Water Level' in self.variables:
            self._load_water_level(start_date=start_date, end_date=end_date)
            datasets.append(self.water_levels['wlevel'])

        # Merge into single dataframe
        data = pd.concat(datasets, axis=1)
        if sum(~data.index.isin(self.data.index)) == 0:
            print('No new data available.')
        else:
            data.index.name = f'time_{self.tz}'
            data.columns = [i for i in self.variables]
            data = pd.concat([self.data,
                              data[data.index.isin(self.data.index) == False]],
                             axis=0)
            self.data = data
            self.filtered_data = \
                pd.concat([self._filter_data(self.data[var],
                                            hr_threshold=self.meta['hr_threshold'],
                                            day_threshold=self.meta['day_threshold'])
                                       for var in self.variables], axis=1)
            self.filtered_data = self._DOY(self.filtered_data)
            statsOutFile = os.path.join(self.outdir,
                                        'observational_data_record.csv.gz')
            self.data.to_csv(statsOutFile, compression='infer')
            if self.verbose:
                print("Updated observational data written to file "\
                      f"'{statsOutFile}'.")
                print("Done! (Don't forget to run Data.update_stats() to update statistics.)")
    
    def update_stats(self):    
        """Calculate new statistics and update if any changes"""
        # Daily stats
        _new_daily_stats = self.daily_stats()
        if _new_daily_stats.equals(self.daily_records):
            if self.verbose:
                print('No new daily records set.')
        else:
            if self.verbose:
                print('Daily stats differ. Updating and saving to file. If new records have been set, they will be printed below.\n')
                self._compare(old=self.daily_stats, new=_new_daily_stats)
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
                print('Monthly stats dicts differ. Updating and saving to file. If new records have been set, they will be printed below.\n')
                self._compare(old=self.monthly_stats,
                              new=_new_monthly_stats)
            self.monthly_records = _new_monthly_stats
            # Write to file
            statsOutFile = os.path.join(self.outdir, 'statistics-monthly.nc')
            self.monthly_records.to_netcdf(statsOutFile, mode='w')
            if self.verbose:
                print(f"\nUpdated monthly observational statistics written to '{statsOutFile}'")

    def _format_date(self, datestr):
        dtdt = pd.to_datetime(datestr)
        return dt.datetime.strftime(dtdt, '%Y%m%d')
    
    def camel(self, text):
        """Convert 'text' to camel case"""
        s = text.replace(',', '').replace("-", " ").replace("_", " ")
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
        through current day.
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
        current day.
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
        through current day.
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
       through current day.
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
        through current day.
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

    def _DOY(self, df):
        """Calculate year day out of 366"""
        # Day of year as integer
        df['YearDay'] = df.index.day_of_year.astype(int)
        # Years that are NOT leap years
        leapInd = [not calendar.isleap(i) for i in df.index.year]
        mask = (leapInd) & (df.index.month > 2)
        # Advance by one day everything after February 28 
        df.loc[mask, 'YearDay'] += 1
        return df

    def _count_missing_hours(self, group, threshold=3):
        """Return True if the number of hours in a day with missing data is 
        less than or equal to 'threshold' and False otherwise.
        """
        missing_hours = group.resample('1h').mean().isna().sum()
        return missing_hours <= threshold

    def _count_missing_days(self, group, threshold=2):
        """Return True if the number of days in a month with missing data 
        is less than or equal to 'theshold' and False otherwise. Two tests 
        are performed: missing data (NaN) and compare to the number of days in the given month.
        """
        try:
            days_in_month = pd.Period(group.index[0].strftime(format='%Y-%m-%d')).days_in_month
            missing_days = group.resample('1D').mean().isna().sum()
            missing_days_flag = missing_days <= threshold
            days_in_month_flag = days_in_month - group.resample('1D').mean().size <= threshold
            return min(missing_days_flag, days_in_month_flag)
        except IndexError:
            pass

    def _filter_data(self, data, hr_threshold=3, day_threshold=2):
        """Filter data to remove days with more than 'hr_threshold' missing
        hours of data and months with more than 'day_threshold' days of missing
        data.
        """
        # Filter out days missing more than <hr_threshold> hours
        filtered = data.groupby(pd.Grouper(freq='1D')).filter(
            lambda x: self._count_missing_hours(group=x, threshold=hr_threshold))
        # Filter out months missing more than <day_threshold> days
        filtered = filtered.groupby(pd.Grouper(freq='1M')).filter(
            lambda x: self._count_missing_days(group=x, threshold=day_threshold))
        return filtered

    def daily_highs(self):
        """Daily highs"""
        return self.filtered_data.groupby(
            pd.Grouper(freq='1D', closed='left', label='left', dropna=True))\
              .max(numeric_only=True)
    
    def daily_lows(self):
        """Daily lows"""
        return self.filtered_data.groupby(
            pd.Grouper(freq='1D', closed='left', label='left', dropna=True))\
              .min(numeric_only=True)

    def daily_avgs(self, decimals=1, true_average=False):
        """Daily averages by calendar day rounded to 'decimals'. If
        'true_average' is True, all measurements from each 24-hour day will be
        used to calculate the average. Otherwise, only the maximum and minimum
        observations are used. Defaults to False (meteorological standard).
        """
        if true_average:
            return self.filtered_data.groupby(
                pd.Grouper(freq='1D', closed='left', label='left', dropna=True))\
                  .mean(numeric_only=True).round(decimals)
        else:
            dailyHighs = self.daily_highs()
            dailyLows = self.daily_lows()
            results = (dailyHighs + dailyLows) / 2
            return results.round(decimals)

    def daily_avg(self, decimals=1, true_average=False):
        """Daily averages rounded to 'decimals'. If 'true_average' is True, all
        measurements from each 24-hour day will be used to calculate the daily
        average. Otherwise, only the maximum and minimum observations are used.
        Defaults to False (meteorological standard).
        """
        dailyAvgs = self.daily_avgs(decimals=decimals,  true_average=true_average)
        dailyAvg = dailyAvgs.groupby('YearDay')\
                            .mean(numeric_only=True).round(decimals)
        dailyAvg.index = dailyAvg.index.astype(int)
        results = xr.DataArray(dailyAvg, dims=['yearday', 'variable'])
        results.name = 'Daily Average'
        return results

    def monthly_highs(self, decimals=1, true_average=False):
        """Monthly highs rounded to 'decimals'. If 'true_average' is True, all
        measurements from each 24-hour day will be used to calculate the daily
        average. Otherwise, only the maximum and minimum observations are used.
        Defaults to False (meteorological standard).
        """
        dailyAvgs = self.daily_avgs(decimals=decimals, true_average=true_average)
        monthHighs = dailyAvgs.groupby(pd.Grouper(freq='M'))\
                              .max(numeric_only=True)
        return monthHighs
    
    def monthly_lows(self, decimals=1, true_average=False):
        """Monthly lows rounded to 'decimals'. If 'true_average' is True, all
        measurements from each 24-hour day will be used to calculate the daily
        average. Otherwise, only the maximum and minimum observations are used.
        Defaults to False (meteorological standard).
        """
        dailyAvgs = self.daily_avgs(decimals=decimals, true_average=true_average)
        monthLows = dailyAvgs.groupby(pd.Grouper(freq='M'))\
                             .min(numeric_only=True)
        return monthLows
    
    def monthly_avg(self, decimals=1, true_average=False):
        """Monthly averages for variable 'var' rounded to 'decimals'. If
        'true_average' is True, all measurements from each 24-hour day will be
        used to calculate the daily average. Otherwise, only the maximum and
        minimum observations are used. Defaults to False (meteorological
        standard).
        """
        dailyAvgs = self.daily_avgs(decimals=decimals, 
                                    true_average=true_average)
        monthlyMeans = dailyAvgs.groupby(pd.Grouper(freq='M'))\
                                .mean(numeric_only=True).round(decimals)
        monthlyMeans.drop('YearDay', axis=1, inplace=True)
        monthlyAvg = monthlyMeans.groupby(monthlyMeans.index.month)\
                                 .mean(numeric_only=True).round(decimals)
        monthlyAvg.index = monthlyAvg.index.astype(int)
        results = xr.DataArray(monthlyAvg, dims=['month', 'variable'])
        results.name = 'Monthly Average'
        return results

    def record_high_daily_avg(self, decimals=1, true_average=False):
        """Record high daily averages rounded to 'decimals'. If 'true_average'
        is True, all measurements from each 24-hour day will be used to
        calculate the daily average. Otherwise, only the maximum and minimum
        observations are used. Defaults to False (meteorological standard).
        """
        # Calculate the records
        dailyAvgs = self.daily_avgs(decimals=decimals, 
                                    true_average=true_average)
        recordHighDailyAvg = \
            dailyAvgs.groupby('YearDay').max(numeric_only=True).round(decimals)
        recordHighDailyAvg.index = recordHighDailyAvg.index.astype(int)
        # Record years
        recordHighDailyAvgYear = dailyAvgs.groupby('YearDay').apply(
            lambda x: x.sort_index().idxmax(numeric_only=True).dt.year)
        recordHighDailyAvgYear.drop('YearDay', axis=1, inplace=True)
        recordHighDailyAvgYear.index = recordHighDailyAvgYear.index.astype(int)
        recordHighDailyAvgYear.columns = [i+' Year' for i in recordHighDailyAvgYear.columns]
        # Create xarray
        results = pd.concat((recordHighDailyAvg, recordHighDailyAvgYear), 
                            axis=1)
        results = xr.DataArray(results, dims=['yearday', 'variable'])
        results.name = 'Record High Daily Average'
        return results

    def record_high_monthly_avg(self, decimals=1, true_average=False):
        """Record high monthly averages rounded to 'decimals'. If
        'true_average' is True, all measurements from each 24-hour day will be
        used to calculate the daily average. Otherwise, only the maximum and
        minimum observations are used. Defaults to False (meteorological
        standard).
        """
        # Calculate the records
        dailyAvgs = self.daily_avgs(decimals=decimals, 
                                    true_average=true_average)
        monthlyAvgs = dailyAvgs.groupby(pd.Grouper(freq='M'))\
                               .mean(numeric_only=True).round(decimals)
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

    def record_low_daily_avg(self, decimals=1, true_average=False):
        """Record low daily averages rounded to 'decimals'.  If 'true_average'
        True, all measurements from each 24-hour day will be used to calculate
        the average. Otherwise, only the maximum and minimum observations are
        used. Defaults to False (meteorological standard)."""
        # Calculate the records
        dailyAvgs = self.daily_avgs(decimals=decimals, 
                                    true_average=true_average)
        recordLowDailyAvg = \
            dailyAvgs.groupby('YearDay').min(numeric_only=True).round(decimals)
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

    def record_low_monthly_avg(self, decimals=1, true_average=False):
        """Record low monthly averages rounded to 'decimals'. If 'true_average'
        is True, all measurements from each 24-hour day will be used to
        calculate the daily average. Otherwise, only the maximum and minimum
        observations are used. Defaults to False (meteorological standard).
        """
        # Calculate the records
        dailyAvgs = self.daily_avgs(decimals=decimals, 
                                    true_average=true_average)
        monthlyAvgs = dailyAvgs.groupby(pd.Grouper(freq='M'))\
                               .mean(numeric_only=True).round(decimals)
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

    def avg_daily_high(self, decimals=1):
        """Average daily highs rounded to 'decimals'."""        
        dailyHighs = self.daily_highs()
        results = dailyHighs.groupby('YearDay')\
                            .mean(numeric_only=True).round(decimals)
        results = xr.DataArray(results, dims=['yearday', 'variable'])
        results.name = 'Average Daily High'
        return results

    def avg_monthly_high(self, decimals=1, true_average=False):
        """Average monthly highs rounded to 'decimals'. If 'true_average' is
        True, all measurements from each 24-hour day will be used to calculate
        the daily average. Otherwise, only the maximum and minimum observations
        are used. Defaults to False (meteorological standard).
        """
        monthlyHighs = self.monthly_highs(decimals=decimals, 
                                          true_average=true_average)
        monthlyHighs.drop('YearDay', axis=1, inplace=True)
        avgMonthlyHighs = monthlyHighs.groupby(monthlyHighs.index.month)\
                                      .mean(numeric_only=True).round(decimals)
        results = xr.DataArray(avgMonthlyHighs, dims=['month', 'variable'])
        results.name = 'Average Monthly High'
        return results

    def lowest_daily_high(self, decimals=1):
        """Lowest daily highs rounded to 'decimals'."""
        # Calculate the record
        dailyHighs = self.daily_highs()
        lowestHigh = dailyHighs.groupby('YearDay')\
                               .min(numeric_only=True).round(decimals)
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

    def lowest_monthly_high(self, decimals=1, true_average=False):
        """Lowest monthly highs rounded to 'decimals'. If 'true_average' is
        True, all measurements from each 24-hour day will be used to calculate
        the daily average. Otherwise, only the maximum and minimum observations
        are used. Defaults to False (meteorological standard).
        """
        # Calculate the record
        monthlyHighs = self.monthly_highs(decimals=decimals, true_average=true_average)
        monthlyHighs.drop('YearDay', axis=1, inplace=True)
        lowMonthlyHigh = monthlyHighs.groupby(monthlyHighs.index.month)\
                                     .min(numeric_only=True).round(decimals)
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

    def record_daily_high(self, decimals=1):
        """Record daily highs rounded to 'decimal'."""
        # Calculate the record
        dailyHighs = self.daily_highs()
        recordHigh = dailyHighs.groupby('YearDay')\
                               .max(numeric_only=True).round(decimals)
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

    def record_monthly_high(self, decimals=1, true_average=False):
        """Record monthly highs rounded to 'decimals'. If 'true_average' is
        True, all measurements from each 24-hour day will be used to calculate
        the daily average. Otherwise, only the maximum and minimum observations
        are used. Defaults to False (meteorological standard).
        """
        # Calculate the record
        monthlyHighs = self.monthly_highs(decimals=decimals, 
                                          true_average=true_average)
        monthlyHighs.drop('YearDay', axis=1, inplace=True)
        recordMonthlyHigh = monthlyHighs.groupby(monthlyHighs.index.month)\
                                        .max(numeric_only=True).round(decimals)
        recordMonthlyHigh.index = recordMonthlyHigh.index.astype(int)
        # Record years
        recordMonthlyHighYear = \
            monthlyHighs.groupby(monthlyHighs.index.month).apply(
                lambda x: x.sort_index().idxmax(numeric_only=True).dt.year)
        recordMonthlyHighYear.index = recordMonthlyHighYear.index.astype(int)
        recordMonthlyHighYear.columns = [i+' Year' for i in recordMonthlyHighYear.columns]
        # Create xarray
        results = pd.concat((recordMonthlyHigh, recordMonthlyHighYear), axis=1)
        results = xr.DataArray(results, dims=['month', 'variable'])
        results.name = 'Record Monthly High'
        return results

    def avg_daily_low(self, decimals=1):
        """Average daily lows rounded to 'decimals'."""        
        dailyLows = self.daily_lows()
        results = dailyLows.groupby('YearDay')\
                           .mean(numeric_only=True).round(decimals)
        results = xr.DataArray(results, dims=['yearday', 'variable'])
        results.name = 'Average Daily Low'
        return results

    def avg_monthly_low(self, decimals=1, true_average=False):
        """Average monthly lows rounded to 'decimals'. If 'true_average' is
        True, all measurements from each 24-hour day will be used to calculate
        the daily average. Otherwise, only the maximum and minimum observations
        are used. Defaults to False (meteorological standard).
        """
        monthlyLows = self.monthly_lows(decimals=decimals, 
                                        true_average=true_average)
        monthlyLows.drop('YearDay', axis=1, inplace=True)
        avgMonthlyLows = monthlyLows.groupby(monthlyLows.index.month)\
                                    .mean(numeric_only=True).round(decimals)
        results = xr.DataArray(avgMonthlyLows, dims=['month', 'variable'])
        results.name = 'Average Monthly Low'
        return results

    def highest_daily_low(self, decimals=1):
        """Highest daily lows rounded to 'decimals'."""
        # Calculate the record
        dailyLows = self.daily_lows()
        highestLow = dailyLows.groupby('YearDay')\
                              .max(numeric_only=True).round(decimals)
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

    def highest_monthly_low(self, decimals=1, true_average=False):
        """Highest monthly lows rounded to 'decimals'. If 'true_average' is
        True, all measurements from each 24-hour day will be used to calculate
        the daily average. Otherwise, only the maximum and minimum observations
        are used. Defaults to False (meteorological standard).
        """
        # Calculate the record
        monthlyLows = self.monthly_lows(decimals=decimals, 
                                        true_average=true_average)
        monthlyLows.drop('YearDay', axis=1, inplace=True)
        highestMonthlyLow = monthlyLows.groupby(monthlyLows.index.month)\
                                       .max(numeric_only=True).round(decimals)
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

    def record_daily_low(self, decimals=1):
        """Record daily lows rounded to 'decimals'."""
        # Calculate the record
        dailyLows = self.daily_lows()
        recordLow = dailyLows.groupby('YearDay')\
                             .min(numeric_only=True).round(decimals)
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

    def record_monthly_low(self, decimals=1, true_average=False):
        """Record monthly lows rounded to 'decimals'. If 'true_average' is
        True, all measurements from each 24-hour day will be used to calculate
        the daily average. Otherwise, only the maximum and minimum observations
        are used. Defaults to False (meteorological standard).
        """
        # Calculate the record
        monthlyLows = self.monthly_lows(decimals=decimals, 
                                        true_average=true_average)
        monthlyLows.drop('YearDay', axis=1, inplace=True)
        recordMonthlyLow = monthlyLows.groupby(monthlyLows.index.month)\
                                      .min(numeric_only=True).round(decimals)
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
            [self.filtered_data[[v, 'YearDay']]\
                .dropna().groupby('YearDay').apply(
                    lambda x: len(x.index.year.unique())) \
                for v in self.filtered_data.columns if v != 'YearDay'], axis=1)
        numYears.columns = [v for v in self.filtered_data.columns \
                            if v != 'YearDay']
        results = xr.DataArray(numYears, dims=['yearday', 'variable'])
        results.name = 'Number of Years'
        return results

    def number_of_years_bymonth(self):
        """Number of years in the historical data records by month."""
        numYears = pd.concat(
            [self.filtered_data[v]\
                .dropna().groupby(self.filtered_data[v].dropna().index.month).apply(
                    lambda x: len(x.index.year.unique())) \
                for v in self.filtered_data.columns if v != 'YearDay'], axis=1)
        numYears.columns = [v for v in self.filtered_data.columns \
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
                     if k not in ['outdir', 'variables', 'units']})
        
        # Add data units for each variable to the array as metadata attributes
        for k, v in self.meta['units'].items():
            daily_records.attrs[k+' units'] = v
        
        # Add time series ranges for each variable to the array as metadata 
        # attributes
        for var in daily_records.coords['variable'].values:
            if 'Year' not in var:
                daily_records.attrs[var+' data range'] = \
                    (self.filtered_data[var].dropna().index.min().strftime('%Y-%m-%d'),
                    self.filtered_data[var].dropna().index.max().strftime('%Y-%m-%d'))
        
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
        day_years.coords['variable'] = [i.removesuffix(' Year') for i in day_years.coords['variable'].values]

        # Merge arrays together
        daily_records = xr.merge([day_records, day_years])
        daily_records = daily_records[
            [item for items in zip(day_records.data_vars, day_years.data_vars) for item in items]]
        daily_records = daily_records.drop_vars(
            [x for x in daily_records.data_vars if daily_records[x].isnull().all()])
        
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
                     if k not in ['outdir', 'variables', 'units']})
        # Add data units for each variable to the array as metadata attributes
        for k, v in self.meta['units'].items():
            monthly_records.attrs[k+' units'] = v

        # Add time series ranges for each variable to the array as metadata 
        # attributes
        for var in monthly_records.coords['variable'].values:
            if 'Year' not in var:
                monthly_records.attrs[var+' data range'] = \
                    (self.filtered_data[var].dropna().index.min().strftime('%Y-%m-%d'),
                    self.filtered_data[var].dropna().index.max().strftime('%Y-%m-%d'))
        
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
        mon_years.coords['variable'] = [i.removesuffix(' Year') for i in mon_years.coords['variable'].values]

        # Merge arrays together
        monthly_records = xr.merge([mon_records, mon_years])
        monthly_records = monthly_records[
            [item for items in zip(mon_records.data_vars, mon_years.data_vars) for item in items]]
        monthly_records = monthly_records.drop_vars(
            [x for x in monthly_records.data_vars if monthly_records[x].isnull().all()])
        
        # Convert years to integers
        monthly_records[[i for i in monthly_records.data_vars if "Year" in i]] = monthly_records[[i for i in monthly_records.data_vars if "Year" in i]].astype(int)

        # Replace yearday with calendar day and rename coordinate
        monthly_records.coords['month'] = \
            pd.date_range(start='2020-01-01', end='2020-12-31', freq='1m')\
              .strftime('%b')
        monthly_records = monthly_records.rename({'month': 'Month'})

        return monthly_records

    def _compare(self, old, new):
        """
        Compare 'old' and 'new' records xarrays excluding daily highs, lows, and averages, since these will always change with updated data
        """
        # Mask out unchanged records
        diffs = old != new
        exclude = ['Daily Average', 'Monthly Average', 'Average High', 'Average Low', 'Years']
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
        # This is a hacky solution that pieces together the various compoents of the 
        # iterrows() output.
        for row in new_records.iterrows():
            for record in record_set:
                if not np.isnan(row[1][record]):
                    var = row[0][1]
                    newRecord = row[1][record]
                    newYear = int(row[1][record+' Year'])
                    try:
                        oldRecord = old.sel(variable=var,
                                            Date=row[0][0])[record].values
                        oldYear = old .sel(variable=var,
                                           Date=row[0][0])[record+' Year']\
                                      .values
                    except KeyError:
                        oldRecord = old.sel(variable=var,
                                            Month=row[0][0])[record].values
                        oldYear = old.sel(variable=var,
                                          Month=row[0][0])[record+' Year']\
                                     .values
                    units = self.units[var]
                    print(f"{record.capitalize()} {var.lower()} set {row[0][0]} {newYear}:\n\t"\
                        f"{newRecord} {units} (previously {oldRecord} {units} in {oldYear})")
    
    def get_daily_stats(self, var=None):
        """Return the daily statistics for variable 'var'"""
        try:
            return self.daily_records.sel(variable=var)
        except AttributeError:
            raise AttributeError('Instance of Data has no daily stats yet. '\
                                 'Run Data.stats() to calculate stats and '\
                                 'try again.')
    
    def get_monthly_stats(self, var=None):
        """Return the monthly statistics dictionary"""
        try:
            return self.monthly_records.sel(variable=var)
        except AttributeError:
            raise AttributeError('Instance of Data has no monthly stats yet. '\
                                 'Run Data.stats() to calculate stats and '\
                                 'try again.')

    def get_daily_stats_table(self, var=None):
        """Return the daily statistics table"""
        try:
            return self.daily_records.sel(variable=var)\
                                     .to_dataframe().drop('variable', axis=1)
        except AttributeError:
            raise AttributeError('Instance of Data has no daily stats table '\
                                 'yet. Run Data.daily_stats() to calculate '\
                                 'stats and try again.')
            
    def get_monthly_stats_table(self, var=None):
        """Return the monthly statistics table"""
        try:
            return self.monthly_records.sel(variable=var)\
                                       .to_dataframe().drop('variable', axis=1)
        except AttributeError:
            raise AttributeError('Instance of Data has no monthly stats '\
                                 'table yet. Run Data.monthly_stats() to '\
                                 'calculate stats and try again.')

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
        return("Oceanic and atmospheric observations for station "\
               f"'{self.name}' (station ID {self.id}):\n"\
               f"{self.station.data_inventory}")
    
    def __repr__(self):
        return(f"{type(self).__name__}("\
               f"stationname='{self.name}', stationid='{self.id}', "\
               f"outdir='{self.outdir}', timezone='{self.tz}', "\
               f"units='{self.units}', datum='{self.datum}', "\
               f"hr_threshold='{self.hr_threshold}', "\
               f"day_threshold='{self.day_threshold}')")