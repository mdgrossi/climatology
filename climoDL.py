# =============================================================================
# climoDL.py 
#
# Author: mdgrossi
# Modified: Oct 11, 2024
#
# This script retrieves NOAA CO-OPS observational data, both atmospheric and
# oceanic, for the specified station. If historical data already exists
# locally, it is updated with the most recently available observations.
#
# To execute for a new station:
# python climoDL.py -s "Virginia Key, FL" -i "8723214" -u "english" -t "lst" -d "MHHW" --hr 3 --day 2 
#
#  NEED TO IMPLEMENT:
#  - Commit and push new data to GitHub to trigger gh-pages
#      OR
#  - Figure out how to run this script daily on GitHub
#    (will need to be able to write data to file)
# 
# =============================================================================
# PACKAGES

from pyclimo import Data
import argparse
import os

# -----------------------------------------------------------------------------
# FUNCTIONS
def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Function control parameters.',
                        prog='climoDL',
                        usage='%(prog)s [arguments]')
    parser.add_argument('-s', '--station', metavar='station', type=str,
                        help='Desired name of station. Used for saving data.',
                        default=None)
    parser.add_argument('-i', '--id', metavar='stationid', type=str,
                        help='Tide station number from which to retrieve data.',
                        default=None)
    parser.add_argument('-o', '--outdir', metavar='outdir', type=str,
                        help='Directory to save data to.',
                        default=None)
    parser.add_argument('-u', '--units', metavar='units', type=str,
                      help='Data units, either "metric" or "english".',
                      default='english')
    parser.add_argument('-t', '--timezone', metavar='timezone', type=str,
                      help='Timezone, either "gmt", "lst", or "lst_ldt".',
                      default='lst')
    parser.add_argument('-d', '--datum', metavar='datum', type=str,
                        help='Tidal datum for water level data. Options: '+
                             '"STND", "MHHW", "MHW", "MTL", "MSL", "MLW", '+
                             '"MLLW", "NAVD"',
                        default='MHHW')
    parser.add_argument('--hr', metavar='hr_threshold', type=int,
                        help='Max number of hours of data that can be missing.',
                        default=3)
    parser.add_argument('--day', metavar='day_threshold', type=int,
                        help='Max number of days of data that can be missing.',
                        default=2)
    parser.add_argument('-r', '--redownload', action='store_true',
                      help='Force redownload of historical data.')
    parser.add_argument('-v', '--verbose', action='store_true',
                      help='Print statuses to screen.')
    return parser.parse_args()

# =============================================================================
# MAIN PROGRAM

def main():
    # Parse command line arguments
    args = parse_args()
    if not args.outdir:
        args.outdir = os.getcwd()

    # Download data
    data = Data(
        stationname=args.station,
        stationid=args.id,
        units=args.units,
        timezone=args.timezone,
        datum=args.datum,
        outdir=args.outdir,
        hr_threshold=args.hr,
        day_threshold=args.day,
        verbose=args.verbose
        )
    data.update_data()
    data.update_stats()

if __name__ == "__main__":
    main()