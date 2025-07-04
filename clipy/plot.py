from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime as dt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from great_tables import GT, loc, style
from bokeh.plotting import figure, show
import bokeh.models as bm
import xarray as xr
import pandas as pd
import numpy as np
import os

# Colors for plotting:
# mg = my colors
# bm = B. McNoldy's colors
# cb = colorblind friendly colors (from https://www.nceas.ucsb.edu/sites/default/files/2022-06/Colorblind%20Safe%20Color%20Schemes.pdf)
colors = dict(
    mg=dict({
        'Date': 'white',
        'Month': 'white',
        'Daily Average': '#F5F5F5',
        'Monthly Average': '#F5F5F5',
        'Record High Daily Average': '#ff8080',
        'Record High Daily Average Year': '#ff8080',
        'Record High Monthly Average': '#ff8080',
        'Record High Monthly Average Year': '#ff8080',
        'Record Low Daily Average': '#c1d5f8',
        'Record Low Daily Average Year': '#c1d5f8',
        'Record Low Monthly Average': '#c1d5f8',
        'Record Low Monthly Average Year': '#c1d5f8',
        'Average High': '#dc8d8d',
        'Lowest High': '#e6aeae',
        'Lowest High Year': '#e6aeae',        
        'Record High': '#d26c6c',
        'Record High Year': '#d26c6c',
        'Average Low': '#a2bff4',
        'Highest Low': '#d1dffa',
        'Highest Low Year': '#d1dffa',
        'Record Low': '#74a0ef',
        'Record Low Year': '#74a0ef',
        'Years': 'white',
        'Plot Light Color': '#D3D3D3'}),
    bm=dict({
        'Date': 'white',
        'Month': 'white',
        'Daily Average': 'gainsboro',
        'Monthly Average': 'gainsboro',
        'Record High Daily Average': 'mistyrose',
        'Record High Daily Average Year': 'mistyrose',
        'Record High Monthly Average': 'mistyrose',
        'Record High Monthly Average Year': 'mistyrose',
        'Record Low Daily Average': 'lavender',
        'Record Low Daily Average Year': 'lavender',
        'Record Low Monthly Average': 'lavender',
        'Record Low Monthly Average Year': 'lavender',
        'Average High': 'orangered',
        'Lowest High': 'darkorange',
        'Lowest High Year': 'darkorange',        
        'Record High': 'orange',
        'Record High Year': 'orange',
        'Average Low': 'mediumpurple',
        'Highest Low': 'navyblue',
        'Highest Low Year': 'navyblue',
        'Record Low': 'lightblue',
        'Record Low Year': 'lightblue',
        'Years': 'white',
        'Plot Light Color': 'white'}),
    cb=dict({
        'Date': '#f9f9f9',
        'Month': '#f9f9f9',
        'Daily Average': '#F5F5F5',
        'Monthly Average': '#F5F5F5',
        'Record High Daily Average': '#d75f4c',
        'Record High Daily Average Year': '#d75f4c',
        'Record High Monthly Average': '#d75f4c',
        'Record High Monthly Average Year': '#d75f4c',
        'Record Low Daily Average': '#3a93c3',
        'Record Low Daily Average Year': '#3a93c3',
        'Record Low Monthly Average': '#3a93c3',
        'Record Low Monthly Average Year': '#3a93c3',
        'Average High': '#f6a482',
        'Lowest High': '#fedbc7',
        'Lowest High Year': '#fedbc7',        
        'Record High': '#b31529',
        'Record High Year': '#b31529',
        'Average Low': '#8ec4de',
        'Highest Low': '#d1e5f0',
        'Highest Low Year': '#d1e5f0',
        'Record Low': '#1065ab',
        'Record Low Year': '#1065ab',
        'Years': '#f9f9f9',
        'Plot Light Color': '#f9f9f9'})
    )

# Helper functions
def record_counts(data, var):
    """Count the number of records set each year for variable `var`

    Parameters
    ----------
    data : xarray
        xarray containing daily or monthly records for a CO-OPS station
    var : str
        Name of the variable to regress. Must be climatology dataset.

    Returns
    -------
    pd.DataFrame
        Pandas dataframe containing value counts of monthly records in each year
    """

    # Dataframe
    stats = data.sel(variable=var).to_dataframe()
    stats = stats[stats.columns[stats.columns.str.endswith('Year')]]

    # Value counts for each year
    counts = stats.stack().groupby(level=[1]).value_counts().unstack().T.fillna(0)
    
    # Resort rows and columns after unstack
    counts = counts.reindex(range(min(counts.index), max(counts.index)+1), fill_value=0)
    counts = counts[stats.columns[stats.columns.str.endswith('Year')]]

    # Restructure for Bokeh plotting
    counts.columns = [i.replace(' Year', '') for i in counts.columns]
    counts.index.name = 'Year'
    counts.index = counts.index.astype(str)
    counts.reset_index(inplace=True)
    return counts

def daily_records_set(data, var, type):
    """Count the number of daily records set each day of each year for variable `var`

    Parameters
    ----------
    data : pyclimo Data object
        Data object containing observational data for a CO-OPS station
    var : str
        Name of the variable to regress. Must be in climatology dataset.
    type : {'daily', 'monthly'}
        Which record file to process (daily or monthly)
    
    Returns
    -------
    pd.DataFrame
        Pandas dataframe containing value counts of daily records in each year
    """

    # Load data
    if type.lower() == 'daily':
        df = xr.load_dataset(os.path.join(data.outdir, 'annual_daily_record_count.nc')).sel(variable=var+' Year').to_dataframe().drop('variable', axis=1)
    elif type.lower == 'monthly':
        df = xr.load_dataset(os.path.join(data.outdir, 'annual_monthly_record_count.nc')).sel(variable=var+' Year').to_dataframe().drop('variable', axis=1)
    else:
        raise ValueError(
            f'`type` must be either "daily" or "monthly" but "{type}" was passed.'
        )
    df.index.name = 'Year'
    df.index = df.index.astype(str)
    df = df[df.sum(axis=1)>0]
    df.reset_index(inplace=True)
    return df

def cos_fit(data):
    """Fit cosine curve to data
    
    Parameters
    ----------
    data : list or 1d array of data to be fit

    Returns
    -------
    Array of fitted values
    """
    X = np.arange(0, len(data))/len(data)

    # Initial parameter values
    guess_freq = 1
    guess_amplitude = 3*np.std(data)/(2**0.5)
    guess_phase = 0
    guess_offset = np.mean(data)
    p0 = [guess_freq, guess_amplitude,
          guess_phase, guess_offset]

    # Function to fit
    def my_cos(x, freq, amplitude, phase, offset):
        return np.cos(x * freq + phase) * amplitude + offset

    # Fit curve to data
    fit = curve_fit(my_cos, X, data, p0=p0)
    
    return my_cos(np.array(X), *fit[0])

def round_down(num, divisor):
    """Round down to the nearest divisor. For example, round_down(45.5, 10)
    will return 40. To round up to the nearest divisor, see `round_up`.

    Parameters
    ----------
    num : float
        Number to be rounded down
    divisor : int
        Divisor of num
    
    Returns
    -------
    Float resulting from `num - (num % divisor)`
    """
    return num - (num%divisor)

def round_up(num, divisor):
    """Round up to the nearest divisor. For example, round_up(45.5, 10) will
    return 50. To round down to the nearest divisor, see `round_down`.

    Parameters
    ----------
    num : float
        Number to be rounded up
    divisor : int
        Divisor of num
    
    Returns
    -------
    Float resulting from `num + (num % divisor)`
    """
    return num + (divisor - (num%divisor))

def getval(stats, var, record):
    """Retrieve `var` `record` from xarraya `stats` for use in valueboxes
    
    Parameters
    ----------
    stats : xarray
        Xarray containing daily or monthly statistics
    var : str
        Variable to be retrieved. Must be in `stats`.
    record : str
        Record to be retrieved. Must be in `stats`.
    """
    deg = u'\N{DEGREE SIGN}'
    val = stats[record].sel(variable=var, Date=dt.today()).values
    return str(val)+f' {deg}F'

# Bokeh plots
def config_plot(p, scheme='cb'):
    """Configure Bokeh plot for website

    Parameters
    ----------
    p : Bokeh Figure class
    scheme : {'mg', 'bm', 'cb}
        Specifies which color scheme to use: 'mg' for M. Grossi's, 'bm' for
        B. McNoldy's, or 'cb' to use a colorblind scheme. Defaults to 'cb'.
    """

    # Plot properties
    p.background_fill_color = '#404040'
    p.border_fill_color = '#404040'
    p.width = 1000
    p.outline_line_color = None
    p.sizing_mode = 'scale_height'

    # x-axis
    p.xgrid.grid_line_color = None
    p.xaxis.axis_line_color = 'grey'
    p.xaxis.major_tick_line_color = 'grey'
        
    # y-axis
    p.yaxis.axis_label_text_color = colors[scheme]['Plot Light Color']
    p.ygrid.grid_line_color = 'grey'
    p.yaxis.axis_line_color = None
    p.yaxis.major_tick_line_color = None
    p.yaxis.minor_tick_line_color = None
    p.outline_line_color = None

    # Fonts
    p.title.text_font = 'arial narrow'
    p.title.text_font_size = '16px'
    p.title.text_color = 'darkgray'
    p.xaxis.major_label_text_font = 'arial narrow'
    p.xaxis.major_label_text_color = colors[scheme]['Plot Light Color']
    p.xaxis.major_label_text_font_size = '14px'
    p.yaxis.major_label_text_font = 'arial narrow'
    p.yaxis.axis_label_text_font = 'arial narrow'
    p.yaxis.axis_label_text_font_style = 'normal'
    p.yaxis.major_label_text_color = colors[scheme]['Plot Light Color']    
    p.yaxis.major_label_text_font_size = '14px'
    p.yaxis.axis_label_text_font_size = '14px'

def histograms(stats, var, y_range=None, scheme='cb'):
    """Plot histograms of record counts per year

    Paramaters
    ----------
    stats : xarray
        xarray containing daily or monthly records for a CO-OPS station,
        arranged with years as rows and record as columns
    var : str
        Name of variable to be plotted. Must be in `stats`.
    y_range : list or tuple of length 2
        List of tuple containing the lower and upper bounds of the y-axis to be displayed. These are create scroll limits for interactivity.
    scheme : {'mg', 'bm', 'cb}
        Specifies which color scheme to use: 'mg' for M. Grossi's, 'bm' for
        B. McNoldy's, or 'cb' to use a colorblind scheme. Defaults to 'cb'.
    """
    data = record_counts(data=stats, var=var)
    # Create histogram for each record
    for col in data.columns:
        if col != 'Year':

            # Create plot
            p = figure(x_range=data['Year'], tooltips="@Year: @$name", height=400,
                       tools='pan, wheel_zoom, box_zoom, undo, reset, fullscreen',
                       title=f'Distribution of {col.lower()}s'.upper())
            bars = p.vbar(x='Year', top=col, source=data,
                          name=col, color=colors[scheme][col], alpha=0.85)
            config_plot(p, scheme=scheme)
            p.title.text_font_size = '20px'

            # x-axis
            p.xaxis.major_label_orientation = 45
            p.x_range.range_padding = 0.05

            # y-axis
            p.yaxis.axis_label='Number of Records Set'
            p.y_range.start = 0
            p.yaxis.axis_label='Number of Records'
            if y_range is not None:
                p.y_range = bm.Range1d(min(y_range), max(y_range),
                                       bounds=(min(y_range), max(y_range)))
            show(p)

def annual_histograms(stats, var, y_range=None, scheme='cb'):
    """Plot histograms of record counts set each year

    Paramaters
    ----------
    stats : xarray
        xarray containing daily or monthly records for a CO-OPS station,
        arranged with years as rows and record as columns
    var : str
        Name of variable to be plotted. Must be in `stats`.
    y_range : list or tuple of length 2
        List of tuple containing the lower and upper bounds of the y-axis to be displayed. These are create scroll limits for interactivity.
    scheme : {'mg', 'bm', 'cb}
        Specifies which color scheme to use: 'mg' for M. Grossi's, 'bm' for
        B. McNoldy's, or 'cb' to use a colorblind scheme. Defaults to 'cb'.
    """

    # Dataframe
    data = stats.sel(variable=var+' Year').to_dataframe().drop('variable', axis=1)
    data.index.name = 'Year'
    data.reset_index(inplace=True)
    data['Year'] = data['Year'].astype(str)

    # Create histogram for each record
    thisYear = dt.today().year
    for col in data.columns:
        if col != 'Year':

            # Create plot
            p = figure(x_range=[str(x) for x in range(1987, thisYear+1)], 
                       tooltips="@Year: @$name", height=400,
                       tools='pan, wheel_zoom, box_zoom, undo, reset, fullscreen',
                       title=f'{col.capitalize()}s set each year')
            bars = p.vbar(x='Year', top=col, source=data,
                        name=col, alpha=0.85,
                        color=colors[scheme][col])
            config_plot(p, scheme=scheme)
            
            # x-axis
            p.xaxis.major_label_orientation = 45
            p.x_range.range_padding = 0.05

            # y-axis
            p.yaxis.axis_label = 'Number of Records Set'
            p.y_range.start = 0
            p.yaxis.axis_label = 'Records Set'
            if y_range is not None:
                p.y_range = bm.Range1d(min(y_range), max(y_range),
                                       bounds=(min(y_range), max(y_range)))
            show(p)

def gtable(stats, var, scheme='cb'):
    """Display a great_tables table if the variable 'var' exists in 'stats'. Otherwise, display a message that the data do not exist.
    
    Parameters
    ----------
    stats : xarray
        Data array containing daily or monthly stats
    var : str
        Variable to retrieve. Must be in `stats`
    scheme : {'mg', 'bm', 'cb}
        Specifies which color scheme to use: 'mg' for M. Grossi's, 'bm' for
        B. McNoldy's, or 'cb' to use a colorblind scheme. Defaults to 'cb'.
    
    Returns
    -------
    Displays a `great_tables` table summarizing stats of variable `var`
    """
    def getrows(record):
        thisyear = dt.today().year
        return df[(df[record] == thisyear)].index.to_list()

    def getcols(record):
        return [record.replace(' Year', ''), record]
    
    # Extract data to dataframe
    freq = 'Monthly' if 'Month' in stats.sizes.keys() else 'Daily'
    try:
        if var == 'Water Level':
            df = stats.sel(variable=var.title()).to_dataframe()\
                      .drop('variable', axis=1).round(2).reset_index()
        else:
            df = stats.sel(variable=var.title()).to_dataframe()\
                      .drop('variable', axis=1).round(1).reset_index()
        
        # Data record
        ts_start = dt.strptime(stats.attrs[f'{var} data range'][0], '%Y-%m-%d').strftime('%-m/%-d/%Y')
        ts_end = dt.strptime(stats.attrs[f'{var} data range'][1], '%Y-%m-%d').strftime('%-m/%-d/%Y')
 
        # Records this year
        thisYear = pd.to_datetime('today').year
        cols = df.columns[df.columns.str.endswith('Year')]
        thisYearRecords = (df==thisYear)[cols].sum().sum()
        lastYearRecords = (df==thisYear-1)[cols].sum().sum()

        # Add columns
        gtbl = GT(df)
        for column in df.columns:
            gtbl = gtbl.tab_style(style=[style.fill(color=colors[scheme][column]), style.text(v_align='middle')], locations=loc.body(columns=column))
    
        # Format table
        gtbl = (gtbl
        .cols_align(align='center')
        .tab_style(style=[style.text(color='gainsboro', weight='bold'), style.fill(color='dimgray')], locations=loc.column_header())
        .tab_options(table_font_size='13px', 
                     table_body_hlines_color='white',
                     heading_align='left',
                     heading_title_font_weight='bold',
                     heading_background_color='gainsboro')
        .tab_header(title="""As of today, {} {} record observations have been reported this year. Last year, {} records were reported.""".format(thisYearRecords, var.lower(), lastYearRecords),
                    subtitle='DATA RECORD: {} - {}'.format(ts_start, ts_end))
        )
        
        # Bolden records this year
        for record in df.columns[df.columns.str.endswith('Year')]:
            gtbl = gtbl.tab_style(
                style = style.text(weight='bold'),
                locations = loc.body(columns=getcols(record), rows=getrows(record)))

        gtbl.show()

    except KeyError:
        print(f'{freq} {var.lower()} data are not available for this station.')

def daily_climo(data, var, flood_thresholds, scheme='cb'):
    """Create a daily climatology plot for environmental variable `var`
    from `data`.
    
    Parameters
    ----------
        data : xarray
            Data array containing climatological stats
        var : str
            One of the available environmental variables in `data`
        flood_threshold : dict
            Flood thresholds to add to water level plot
        scheme : {'mg', 'bm', 'cb}
            Specifies which color scheme to use: 'mg' for M. Grossi's, 'bm' for
            B. McNoldy's, or 'cb' to use a colorblind scheme. Defaults to 'cb'.
    """

    # Dates for x axis
    df = data.sel(variable=var).to_dataframe().drop('variable', axis=1)
    df['xdates'] = pd.date_range(start='2020-01-01', end='2020-12-31', freq='1D')
    df['Average High Curve'] = cos_fit(df['Average High'])
    df['Daily Average Curve'] = cos_fit(df['Daily Average'])
    df['Average Low Curve'] = cos_fit(df['Average Low'])
    
    # Records this year
    thisYear = pd.to_datetime('today').year
    df['High Records'] = df['Record High'].where(df['Record High Year'] == thisYear)
    df['Low Records'] = df['Record Low'].where(df['Record Low Year'] == thisYear)
    source = bm.ColumnDataSource(df)
    
    # Create a new plot
    ts_start = dt.strptime(data.attrs[f'{var} data range'][0], '%Y-%m-%d').strftime('%-m/%-d/%Y')
    ts_end = dt.strptime(data.attrs[f'{var} data range'][1], '%Y-%m-%d').strftime('%-m/%-d/%Y')
    p = figure(title=f'DATA RECORD: {ts_start} - {ts_end}',
               x_axis_type='datetime', height=600,
               y_range=(round_down(df['Record Low'].min(), 10),
                        round_up(df['Record High'].max(), 10)),
               tools='pan, wheel_zoom, box_zoom, undo, reset, fullscreen')

    # This year record highs
    hr = p.scatter(x='xdates', y='High Records', source=source,
                   name=f'{thisYear} High Record', size=4, color='white')
    hr.level = 'overlay'
    # This year record lows
    lr = p.scatter(x='xdates', y='Low Records', source=source,
                   name=f'{thisYear} Low Record', size=4, color='white')
    lr.level = 'overlay'
    # Record highs
    rh = p.scatter(x='xdates', y='Record High', source=source,
                   name='Record High', size=2,
                   color=colors[scheme]['Record High'])
    # Average high
    ah = p.line(x='xdates', y='Average High Curve', source=source,
                name='Average High', width=3,
                color=colors[scheme]['Average High'])
    # Daily average
    da = p.line(x='xdates', y='Daily Average Curve', source=source,
                name='Daily Average', width=2,
                color=colors[scheme]['Daily Average'])
    # Average lows
    al = p.line(x='xdates', y='Average Low Curve', source=source,
                name='Average Low', width=3,
                color=colors[scheme]['Average Low'])
    # Record lows
    rl = p.scatter(x='xdates', y='Record Low', source=source,
                   name='Record Low', size=2,
                   color=colors[scheme]['Record Low'],
                   hover_fill_color='white', hover_alpha=0.5)
    config_plot(p)

    # Flood thresholds (water level plot only)
    if var=='Water Level':
        for level, threshold in flood_thresholds.items():
            hline = bm.Span(location=threshold, dimension='width',
                         line_dash=[20,8], line_alpha=0.75,
                         line_color='cadetblue', line_width=2)
            p.renderers.extend([hline])
            mytext = bm.Label(x=pd.to_datetime('2019-12-15'), y=threshold+0.1,
                              text=f'{level} flood threshold'.upper(), text_color='cadetblue',
                              text_font_size='8px',
                              text_font='arial narrow')
            p.add_layout(mytext)
    
    # Tools
    crosshair = bm.CrosshairTool(dimensions='height',
                              line_color='grey', line_alpha=0.5)
    hover = bm.HoverTool(mode='vline', renderers=[da],
                      formatters={'@xdates': 'datetime'})
    units = data.attrs[f"{var} units"]
    if var == 'Water Level':
        hover.tooltips = """
            <b> @xdates{{%b %d}} </b> <br>
            Record High: @{{Record High}}{{0.00}} {u}<br>
            Average High: @{{Average High Curve}}{{0.00}} {u}<br>
            Daily Average: @{{Daily Average Curve}}{{0.00}} {u}<br>
            Average Low: @{{Average Low Curve}}{{0.00}} {u}<br>
            Record Low: @{{Record Low}}{{0.00}} {u}<br>
            {y} High Record: @{{High Records}}{{0.00}} {u}<br>
            {y} Low Record: @{{Low Records}}{{0.00}} {u}
            """.format(u=units, y=thisYear)
    else:
        hover.tooltips = """
            <b> @xdates{{%b %d}} </b> <br>
            Record High: @{{Record High}}{{0.0}} {u}<br>
            Average High: @{{Average High Curve}}{{0.0}} {u}<br>
            Daily Average: @{{Daily Average Curve}}{{0.0}} {u}<br>
            Average Low: @{{Average Low Curve}}{{0.0}} {u}<br>
            Record Low: @{{Record Low}}{{0.0}} {u}<br>
            {y} High Record: @{{High Records}}{{0.0}} {u}<br>
            {y} Low Record: @{{Low Records}}{{0.0}} {u}
            """.format(u=units, y=thisYear)
    p.add_tools(hover, crosshair)
    p.toolbar.autohide = True

    # x-axis
    p.xaxis[0].formatter = bm.DatetimeTickFormatter(months="%b %d")
    p.xaxis[0].ticker.desired_num_ticks = 12
    
    # y-axis
    if var == 'Water Level':
        p.yaxis.axis_label=f'{var} relative to {data.attrs["datum"].upper()} ({data.attrs[f"{var} units"]})'
    else:
        p.yaxis.axis_label=f'{var} ({data.attrs[f"{var} units"]})'
    ymin = round_down(df['Record Low'].min(), 10)
    ymax = round_up(df['Record High'].max(), 10)
    p.y_range = bm.Range1d(ymin, ymax, bounds=(ymin, ymax))
    
    # Legend
    legend = bm.Legend(items=[
        ('{} Record'.format(thisYear), [hr, lr]),
        ('Record High', [rh]),
        ('Average High', [ah]),
        ('Daily Average', [da]),
        ('Average Low', [al]),
        ('Record Low', [rl])],
                    background_fill_color='#404040', border_line_color=None,
                    label_text_color=colors[scheme]['Plot Light Color'],
                    location='center_right', click_policy='mute')
    p.add_layout(legend, 'right')
    show(p)

def monthly_climo(data, var, scheme='cb'):
    """Create a monthly climatology plot for environmental variable `var`
    from `data`.
    
    Parameters
    ----------
        data : xarray
            Data array containing climatological stats
        var : str
            One of the available environmental variables in `data`
        flood_threshold : dict
            Flood thresholds to add to water level plot
        scheme : {'mg', 'bm', 'cb}
            Specifies which color scheme to use: 'mg' for M. Grossi's, 'bm' for
            B. McNoldy's, or 'cb' to use a colorblind scheme. Defaults to 'cb'.
    """

    # Dates for x axis
    df = data.sel(variable=var).to_dataframe().drop('variable', axis=1).reset_index()
    df['Average High Curve'] = cos_fit(df['Average High'])
    df['Monthly Average Curve'] = cos_fit(df['Monthly Average'])
    df['Average Low Curve'] = cos_fit(df['Average Low'])
    
    # Record this year
    thisYear = pd.to_datetime('today').year
    df['High Records'] = df['Record High'].where(df['Record High Year'] == thisYear)
    df['Low Records'] = df['Record Low'].where(df['Record Low Year'] == thisYear)
    source = bm.ColumnDataSource(df)
    
    # Create a new plot
    ts_start = dt.strptime(data.attrs[f'{var} data range'][0], '%Y-%m-%d').strftime('%-m/%-d/%Y')
    ts_end = dt.strptime(data.attrs[f'{var} data range'][1], '%Y-%m-%d').strftime('%-m/%-d/%Y')
    p = figure(title=f'DATA RECORD: {ts_start} - {ts_end}', height=600,
               x_range=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
               y_range=(round_down(df['Record Low'].min(), 1), round_up(df['Record High'].max(), 1)),
               tools='pan, wheel_zoom, box_zoom, undo, reset, fullscreen')

    # This year record highs
    hr = p.scatter(x='Month', y='High Records', source=source,
                   name=f'{thisYear} High Record', size=7, color='white')
    hr.level = 'overlay'
    # This year record lows
    lr = p.scatter(x='Month', y='Low Records', source=source,
                   name=f'{thisYear} Low Record', size=7, color='white')
    lr.level = 'overlay'
    # Record highs
    rh = p.scatter(x='Month', y='Record High', source=source,
                   name='Record High', size=7,
                   color=colors[scheme]['Record High'])
    # Average high
    ah = p.line(x='Month', y='Average High Curve', source=source,
                name='Average High', width=4,
                color=colors[scheme]['Average High'])
    # Monthly average
    ma = p.line(x='Month', y='Monthly Average Curve', source=source,
                name='Monthly Average', width=3,
                color=colors[scheme]['Monthly Average'])
    # Average lows
    al = p.line(x='Month', y='Average Low Curve', source=source,
                name='Average Low', width=4,
                color=colors[scheme]['Average Low'])
    # Record lows
    rl = p.scatter(x='Month', y='Record Low', source=source,
                   name='Record Low', size=7,
                   color=colors[scheme]['Record Low'],
                   hover_fill_color='white', hover_alpha=0.5)
    config_plot(p)
    
    # Tools
    crosshair = bm.CrosshairTool(dimensions='height',
                              line_color='grey', line_alpha=0.5)
    hover = bm.HoverTool(mode='vline', renderers=[ma],
                      formatters={'@xdates': 'datetime'})
    units = data.attrs[f"{var} units"]
    if var == 'Water Level':
        hover.tooltips = """
            <b> @Month </b> <br>
            Record High: @{{Record High}}{{0.00}} {u}<br>
            Average High: @{{Average High Curve}}{{0.00}} {u}<br>
            Monthly Average: @{{Monthly Average Curve}}{{0.00}} {u}<br>
            Average Low: @{{Average Low Curve}}{{0.00}} {u}<br>
            Record Low: @{{Record Low}}{{0.00}} {u}<br>
            {y} High Record: @{{High Records}}{{0.00}} {u}<br>
            {y} Low Record: @{{Low Records}}{{0.00}} {u}
            """.format(u=units, y=thisYear)
    else:
        hover.tooltips = """
            <b> @Month </b> <br>
            Record High: @{{Record High}}{{0.0}} {u}<br>
            Average High: @{{Average High Curve}}{{0.0}} {u}<br>
            Monthly Average: @{{Monthly Average Curve}}{{0.0}} {u}<br>
            Average Low: @{{Average Low Curve}}{{0.0}} {u}<br>
            Record Low: @{{Record Low}}{{0.0}} {u}<br>
            {y} High Record: @{{High Records}}{{0.0}} {u}<br>
            {y} Low Record: @{{Low Records}}{{0.0}} {u}
            """.format(u=units, y=thisYear)
    p.add_tools(hover, crosshair)
    p.toolbar.autohide = True
    
    # y-axis
    if var == 'Water Level':
        p.yaxis.axis_label=f'{var} relative to {data.attrs["datum"].upper()} ({data.attrs[f"{var} units"]})'
    else:
        p.yaxis.axis_label=f'{var} ({data.attrs[f"{var} units"]})'
    
    # Legend
    legend = bm.Legend(items=[
        ('{} Record'.format(thisYear), [hr, lr]),
        ('Record High', [rh]),
        ('Average High', [ah]),
        ('Monthly Average', [ma]),
        ('Average Low', [al]),
        ('Record Low', [rl])],
                    background_fill_color='#404040', border_line_color=None,
                    label_text_color=colors[scheme]['Plot Light Color'],
                    location='center_right', click_policy='mute')
    p.add_layout(legend, 'right')
    show(p)

def trend(data, var, scheme='cb', true_average=False, fname=None):
    """Plot time series trend

    Parameters
    ----------
    data : pyclimo Data object
        Data object containing observational data for a CO-OPS station
    var : str
        Name of the variable to regress. Must be in climatology dataset.
    scheme : {'mg', 'bm', 'cb}
        Specifies which color scheme to use: 'mg' for M. Grossi's, 'bm' for
        B. McNoldy's, or 'cb' to use a colorblind scheme. Defaults to 'mg'.
    true_average : Bool
        If True, all measurements from each 24-hour day will be used to calculate the
        average. Otherwise, only the maximum and minimum observations are used. Defaults to False (meteorological standard).
    fname : str or None
        File name with directory to be written out, if provided. If None, plot will be displayed instead. Defaults to None.
    """
 
    # Monthly averages
    dailyAvgs = data.mon_daily_avgs(true_average=true_average)[var]
    monthlyAvgs = dailyAvgs.groupby(pd.Grouper(freq='M')).mean(numeric_only=True)
    df = pd.DataFrame(monthlyAvgs)
    df = df.loc[df.first_valid_index():df.last_valid_index()]
    
    # Linearly interpret missing data
    df['rownum'] = np.arange(df.shape[0])
    col = df.columns.values[0]
    df_nona = df.dropna(subset = [col])
    f = interp1d(df_nona['rownum'], df_nona[col])
    df['linear_fill'] = f(df['rownum'])

    # Normalize time series to deseasonalize
    tsMin = df['linear_fill'].min()
    tsMax = df['linear_fill'].max()
    tseries_norm = (df['linear_fill'] - tsMin) / (tsMax - tsMin)

    # Deseasonalize time series and un-normalize
    components = seasonal_decompose(tseries_norm, model='additive', period=4)
    deseasoned = tseries_norm - components.seasonal
    df['deseasoned'] = (deseasoned * (tsMax - tsMin)) + tsMin

    # Apply linear regression
    coef = np.polyfit(df['rownum'], df['deseasoned'], 1)
    slope = coef[0] # height/mon
    poly1d_fn = np.poly1d(coef)
    df['linear_reg'] = poly1d_fn(df['rownum'])
    mask = ~df[var].isna()
    masked = np.ma.masked_where(df[var].isna(), df['deseasoned'])
    
    # Create plot
    df.index.name = 'xdates'
    ts_start = df.index.min().strftime('%-m/%-d/%Y')
    ts_end = df.index.max().strftime('%-m/%-d/%Y')
    source = bm.ColumnDataSource(df.reset_index())
    p = figure(title=f'RELATIVE {var.upper()} TREND: {ts_start} - {ts_end}\n'+
                     f'{np.round(slope*12, 4)} {data.units[var]}/yr or {np.round(slope*12*100, 2)} {data.units[var]} in 100 years',
                background_fill_color='#404040', border_fill_color='#404040',
                width=1000, height=400, x_axis_type='datetime',
                tools='pan, wheel_zoom, box_zoom, undo, reset, fullscreen',
                outline_line_color=None, sizing_mode='scale_height')

    # Data with regression line
    sct = p.line(x='xdates', y=var, source=source, name='Monthly Average',
                color=colors['mg']['Plot Light Color'], alpha=0.5)
    sct.level = 'overlay'
    reg = p.line(x='xdates', y='linear_reg', source=source, name='Trend',
                color=colors['mg']['Record Low'], line_width=5)
    reg.level = 'overlay'
    config_plot(p)

    # Tools
    crosshair = bm.CrosshairTool(dimensions='height',
                                line_color='grey', line_alpha=0.5)
    hover = bm.HoverTool(mode='vline', renderers=[sct],
                        formatters={'@xdates': 'datetime'})
    hover.tooltips = f"""
    <b> @xdates{{%b %Y}} </b> <br>
    Monthly Average: @{{Water Level}}{{0.00}} {data.units[var]}
    """
    p.add_tools(hover, crosshair)
    p.toolbar.autohide = True

    # y-axis
    p.yaxis.axis_label = f'Monthly average {var.lower()} relative to {data.datum}'
    if max(df[var]) < 0:
        p.y_range = bm.Range1d(-max(abs(df[var]))-0.5, 0.45,
                                bounds=(-max(abs(df[var]))-0.5, 0.45))
    
    # Save or display
    if fname is not None:
        from bokeh.io import output_file
        from bokeh.plotting import save
        output_file(fname)
        save(p)
    else:
        show(p)
