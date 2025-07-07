# =============================================================================
# climoDL.py 
#
# Author: mdgrossi
# Modified: Oct 11, 2024
#
# This script retrieves NOAA CO-OPS observational data, both atmospheric and
# oceanic, for the specified station. If historical data already exist
# locally, they are updated with the most recently available observations.
#
# To execute for a new station:
# python3 climoDL.py -s "Virginia Key, FL" -i "8723214" -u "english" -t "lst" -d "MHHW" --hr 3 --day 2 
#
#  NEED TO IMPLEMENT:
#  - Capture output and send email
# =============================================================================
# PACKAGES

from clipy import climo, plot
import argparse
import logging
import json
import os

# -----------------------------------------------------------------------------
# FUNCTIONS

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        prog='climoDL',
        usage='%(prog)s [arguments]',
        description='Function control parameters.')
    parser.add_argument(
        '-s', '--station',
        metavar='station', type=str,
        default=None,
        help='Desired name of station. Used for saving data.')
    parser.add_argument(
        '-i', '--id',
        metavar='stationid', type=str,
        default='auto',
        help='Tide station number from which to retrieve data.')
    parser.add_argument(
        '-u', '--units',
        metavar='units', type=str,
        default='english',
        help='Data units, either "metric" or "english".')
    parser.add_argument(
        '-t', '--timezone',
        metavar='timezone', type=str,
        default='lst',
        help='Timezone, either "gmt", "lst", or "lst_ldt".')
    parser.add_argument(
        '-d', '--datum',
        metavar='datum', type=str,
        default='MHHW',
        help='Tidal datum for water level data. Options: '+
             '"STND", "MHHW", "MHW", "MTL", "MSL", "MLW", "MLLW", "NAVD"')
    parser.add_argument(
        '--hr',
        metavar='hr_threshold', type=int,
        default=4,
        help='Max number of hours of data that can be missing.')
    parser.add_argument(
        '--day',
        metavar='day_threshold', type=int,
        default=2,
        help='Max number of days of data that can be missing.')
    parser.add_argument(
        '-r', '--redownload',
        action='store_true',
        help='Force redownload of historical data.')
    parser.add_argument(
        '-p', '--reprocess',
        action='store_true',
        help='Reprocess existing data using new parameters.')
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Print statuses to screen.')
    return parser.parse_args()

# =============================================================================
# MAIN PROGRAM

def main():
    
    # Log errors
    logging.basicConfig(level=logging.ERROR)

    # Parse command line arguments
    args = parse_args()
    
    # Dictionary of station IDs
    # stationids = {
    #     'Beaufort, NC': '8656483',
    #     'Woods Hole, MA': '8447930',
    #     'Naples, FL': '8725114',
    #     'Bay St. Louis, MS': '8747437',
    #     'Virginia Key, FL': '8723214',
    #     'Lewes, DE': '8557380'
    # }
    with open('stations.json', 'r') as sf:
        stationids = json.load(sf)
    if args.id == 'auto':
        try:
            args.id = stationids[args.station]
        except KeyError:
            print('No station ID was passed and unable to automatically determine one using "station". Pass a valid "station" or specify station ID and try again.')

    # Download data and update stats
    data = climo.Data(
        stationname=args.station,
        stationid=args.id,
        units=args.units,
        timezone=args.timezone,
        datum=args.datum,
        hr_threshold=args.hr,
        day_threshold=args.day,
        redownload=args.redownload,
        reprocess=args.reprocess,
        verbose=args.verbose
        )
    try:
        data.update_data()
        data.update_stats()

        # Water Level trend plot
        if 'Water Level' in data.variables:
            plot.trend(data=data, var='Water Level',
                    fname=os.path.join(data.outdir, 'trend-waterlevel.html'))
    except Exception:
        logging.exception(f'An error occurred while updating {args.station}.')

if __name__ == "__main__":
    """Main program"""
    main()