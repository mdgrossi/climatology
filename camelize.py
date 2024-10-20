# =============================================================================
# camelize.py
#
# Author: mdgrossi
# Modified: Oct 18, 2024
#
# A simple utility to return a camelized version of a "City, ST" string with
# special characters removed.
#
# To execute:
# python3 camelize <"City, ST">
# =============================================================================

import argparse

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        prog='camelize',
        usage='%(prog)s [arguments]',
        description='Function control parameters.')
    parser.add_argument('string',
        metavar='string', type=str,
        default=None)
    return parser.parse_args()

def camel(text):
    """Convert 'text' to camel case"""
    s = text.replace(',', '').replace("-", " ").replace("_", " ")
    s = s.split()
    if len(text) == 0:
        return text
    return s[0].lower() + ''.join(i.capitalize() for i in s[1:])

def main():
    # Parse command line arguments
    args = parse_args()
    print(camel(args.string))

if __name__ == "__main__":
    """Main program"""
    main()