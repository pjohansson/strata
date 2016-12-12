import click
from click import argument as add_argument
from click import option as add_option
from click import group as create_group
import numpy as np
import pkg_resources
import sys

from strata.average import average
from strata.convert import convert
from strata.interface.angle import interface_contact_angle
from strata.interface.collect import collect_interfaces
from strata.interface.view import view_interfaces
from strata.interface.area import integrate_interfaces
from strata.contact_line_analysis import extract_contact_line_bins
from strata.spreading.fit import fit_spreading_data
from strata.spreading.collect import collect
from strata.spreading.view import view_spreading
from strata.view_flowmap import view_flowmap_2d, view_flowfields
from strata.sample_average import sample_average_files


# Construct version string
try:
    version = pkg_resources.require("strata")[0].version
except Exception:
    version = "Unknown"


class OptIntParamType(click.ParamType):
    """Either an integer or None."""

    name = 'int/none'

    def convert(self, value, param, ctx):
        try:
            return int(value)
        except ValueError:
            if value.lower() == 'none':
                return None

        self.fail("%r is not an integer or 'None'" % value)

class OptFloatParamType(click.ParamType):
    """Either a float number or None."""

    name = 'float/none'

    def convert(self, value, param, ctx):
        try:
            return float(value)
        except ValueError:
            if value.lower() == 'none':
                return None

        self.fail("%r is not a float number or 'None'" % value)

class ListFloatValsParamType(click.ParamType):
    """A string of float values."""

    name = '"float ..."'

    def convert(self, value, param, ctx):
        try:
            return [float(v) for v in value.split()]
        except Exception:
            self.fail("%r is not a valid string of float numbers" % value)

class ListStrsParamType(click.ParamType):
    """A string of strings."""

    name = '"text ..."'

    def convert(self, value, param, ctx):
        try:
            return [v for v in value.split()]
        except Exception:
            self.fail("%r is not a valid string of float numbers" % value)

OPT_INT = OptIntParamType()
OPT_FLOAT = OptFloatParamType()
STR_FLOATS = ListFloatValsParamType()
STR_LIST = ListStrsParamType()

"""Command line utility to invoke the functionality of strata."""

def print_version(ctx, param, value):
    if not value or ctx.resilient_parsing:
        return

    click.echo('Version %s' % version)
    ctx.exit()

# Main functionality
@create_group()
@click.option('--version', is_flag=True, callback=print_version,
              expose_value=False, is_eager=True)
def strata():
    """Tools for reading and analysing files of flow data."""
    pass


# Description of commands
cmd_average = {
        'name': 'average',
        'desc': 'Average multiple input data files or bins.'
        }
cmd_convert = {
        'name': 'convert',
        'desc': 'Convert data files to another format.'
        }
cmd_contactline = {
        'name': 'contact_line',
        'desc': 'Analyse data around the contact line.'
        }
cmd_sample = {
        'name': 'sample',
        'desc': 'Sample data maps.'
}


cmd_interface = {
        'name': 'interface',
        'desc': 'View, average or collect interface data.'
        }
cmd_intangle = {
        'name': 'angle',
        'desc': 'Measure the contact angle of collected interface data.'
        }
cmd_intcollect = {
        'name': 'collect',
        'desc': 'Find the interface boundary of droplet data.'
        }
cmd_intview = {
        'name': 'view',
        'desc': 'View and average collected interface data.'
}
cmd_intarea = {
        'name': 'area',
        'desc': 'Calculate the integrated area of interfaces.'
}


cmd_spreading = {
        'name': 'spreading',
        'desc': 'View or collect spreading data.'
        }
cmd_collect = {
        'name': 'collect',
        'desc': 'Collect the spreading radius per time for a droplet.'
        }
cmd_view = {
        'name': 'view',
        'desc': 'View data of spreading files.'
        }
cmd_sprfit = {
        'name': 'fit',
        'desc': 'Fit spreading data.'
}


cmd_flowview = {
        'name': 'view',
        'desc': 'View flow field data.'
}
cmd_heightmap = {
        'name': 'height',
        'desc': 'View height map data.'
}
cmd_contourmap = {
        'name': 'contour',
        'desc': 'Draw a contour map.'
}
cmd_quiver = {
        'name': 'flow',
        'desc': 'Visualise flow fields.'
}


# Average wrapper
@strata.command(name=cmd_average['name'], short_help=cmd_average['desc'])
@add_argument('base', type=str)
@add_argument('output', type=str)
@add_argument('group', type=click.IntRange(1, None))
@add_option('--combine',
        type=OPT_INT, nargs=2, default=[None, None],
        help='Combine bins of the system along x and y. (None)')
@add_option('--recenter',
        type=click.Choice(['none', 'left', 'right', 'com']), default='none',
        help='Recenter data maps around a contact line or the center of mass. (none)')
@add_option('--floor', type=float, default=None,
        help='Height to determine the contact line at. (None)')
@add_option('-co', '--cutoff', type=float, default=0.,
        help='Boundary bins require this much mass. (0)')
@add_option('-cb', '--cutoff_bins', default=1,
        help='Boundary bins require this many neighbours.')
@add_option('-cr', '--cutoff_radius', default=1.,
        help='Boundary bins search for neighbours within this radius. (1 nm)')
@add_option('-b', '--begin', default=1,
        type=click.IntRange(0, None), metavar='INTEGER',
        help='Begin reading from BASE at this number. (1)')
@add_option('-e', '--end', default=None,
        type=click.IntRange(0, None), metavar='INTEGER',
        help='End reading from BASE at this number. (None)')
@add_option('--ext', default='.dat',
        help='Read and write using this file extension. (.dat)')
def average_cli(base, output, group, **kwargs):
    """Sample average files at BASE path in bundles of size GROUP
    and write to files at OUTPUT base.

    Optionally the bins of all maps can be combined along x and y
    by supplying the keyword argument `combine`.

    File names are generated by joining the base path and extension with
    a five-digit integer signifying file number ('%s%05d%s').

    By supplying the keyword argument `recenter` the data maps can be
    recentered to either of the left and right contact lines or the
    center of mass of the system before calculating the average. This
    can partly compensate for the movement of the contact line if
    quantities around it is to be analysed. Keep in mind that the
    geometry of the contact line will not be affected by this recentering.

    The determination of the contact line is by the same method as
    in `strata spreading collect`.

    """

    set_none_to_inf(kwargs)

    # Set correct option for recenter
    if kwargs['recenter'] == 'none':
        kwargs['recenter'] = None

    average(base, output, group, **kwargs)


# Convert wrapper
@strata.command(name=cmd_convert['name'], short_help=cmd_convert['desc'])
@add_argument('base', type=str)
@add_argument('output', type=str)
@add_option('--ftype',
        type=click.Choice(['simple', 'simple_plain']), default='simple',
        help='Format to convert files into. (simple)')
@add_option('-b', '--begin', default=1,
        type=click.IntRange(0, None), metavar='INTEGER',
        help='Begin reading from BASE at this number. (1)')
@add_option('-e', '--end', default=None,
        type=click.IntRange(0, None), metavar='INTEGER',
        help='End reading from BASE at this number. (None)')
@add_option('--ext', default='.dat',
        help='Read and write using this file extension. (.dat)')
def convert_cli(base, output, **kwargs):
    """Convert files at BASE path to another data file format
    and write to files at OUTPUT base.

    File names are generated by joining the base path and extension with
    a five-digit integer signifying file number ('%s%05d%s').

    """

    set_none_to_inf(kwargs)
    convert(base, output, **kwargs)


# Combined spreading tools for collection and plotting
@strata.group()
def spreading(name=cmd_spreading['name'], short_help=cmd_spreading['desc']):
    """View or collect spreading data of droplets."""
    pass


# Spreading wrapper
@spreading.command(name=cmd_collect['name'], short_help=cmd_collect['desc'])
@add_argument('base', type=str)
@add_argument('floor', type=float)
@add_option('-o', '--output', type=click.Path(), default=None,
        help='Write the collected data to disk.')
@add_option('-dt', '--delta_t', 'dt', default=1.,
    help='Time difference between data map files. (1)')
@add_option('-co', '--cutoff', type=float, default=None,
        help='Boundary bins require this much mass. (0)')
@add_option('-cr', '--cutoff_radius', default=1.,
        help='Boundary bins search for neighbours within this radius. (1 nm)')
@add_option('-cb', '--cutoff_bins', default=1,
        help='Boundary bins require this many neighbours.')
@add_option('-t0', '--time_init', 't0', type=float, default=0.,
        help='Initial time of first spreading frame (ps)')
@add_option('-b', '--begin', default=1,
        type=click.IntRange(0, None), metavar='INTEGER',
        help='Begin reading from BASE at this number. (1)')
@add_option('-e', '--end', default=None,
        type=click.IntRange(0, None), metavar='INTEGER',
        help='End reading from BASE at this number. (None)')
@add_option('--ext', default='.dat',
        help='Read using this file extension. (.dat)')
@add_option('-v', '--verbose', default=False, is_flag=True,
        help='Verbose output: Print spreading to stdout.')
def spreading_collect_cli(base, floor, **kwargs):
    """Collect spreading radius r(t) at height FLOOR for input files at BASE.

    The radius is calculated by finding the outermost bins fulfilling set
    criteria to be considered parts of the droplet. For each considered bin
    in the bottom-most layer which has more mass than a set cut-off, a
    search is made for similarly filled bins within a set radius. If the
    number of filled bins within this radius surpasses the final requirement,
    the bin is considered to be connected to the main droplet. The left- and
    rightmost of these bins in the selected layer are taken as the droplet
    spreading edges from which the radius is calculated.

    Read data files must have data fields corresponding to coordinates
    and mass.

    File names are generated by joining the base path and extension with
    a five-digit integer signifying file number ('%s%05d%s').

    """

    verbose = kwargs.pop('verbose')

    set_none_to_inf(kwargs)
    kwargs['floor'] = floor
    data = collect(base, **kwargs)

    if verbose:
        print("Time (ps) Radius (nm)")
        for time, radius in data:
            print("%.3f %.3f" % (time, radius))


# Plotting wrapper
@spreading.command(name=cmd_view['name'], short_help=cmd_view['desc'])
@add_argument('files', type=click.Path(exists=True), nargs=-1)
@add_option('-t', '--tau', type=float, default=1., metavar='TAU',
        help='Time scaling factor.')
@add_option('-r', '--radius', 'R', type=float, default=1., metavar='R',
        help='Radius scaling factor.')
@add_option('-rs', '--sync_radius', type=float, default=None,
        help='Synchronise data at this radius. Done after scaling by R. (None)')
@add_option('-o', '--save_fig', type=click.Path(), default=None,
        help='Save figure to path. (None)')
@add_option('-x', '--save_xvg', type=click.Path(), default=None,
        help='Save read data to path. (None)')
@add_option('--show/--noshow', default=True,
        help='Whether or not to draw graph. (True)')
@add_option('--loglog', is_flag=True,
        help='Scale graph axes logarithmically.')
@add_option('--xlim', type=OPT_FLOAT, nargs=2, default=(None, None),
        metavar='MIN MAX', help='Set limits on the x axis.')
@add_option('--ylim', type=OPT_FLOAT, nargs=2, default=(None, None),
        metavar='MIN MAX', help='Set limits on the y axis.')
@add_option('--title', default='Droplet spreading',
        help='Figure title.')
@add_option('--xlabel', default='Time (ps)',
        help='Label of x axis.')
@add_option('--ylabel', default='Radius (nm)',
        help='Label of y axis.')
def spreading_view_cli(files, **kwargs):
    """View spreading data of input FILES.

    Input files must be in whitespace separated format, the first column
    designating time with all following being their corresponding spreading
    radii.

    The combined data can be saved to disk in an XmGrace compatible format.

    """

    view_spreading(files, **kwargs)

# Spreading fit wrapper
@spreading.command(name=cmd_sprfit['name'], short_help=cmd_sprfit['desc'])
@add_argument('files', type=click.Path(exists=True), nargs=-1)
@add_option('--lims', type=OPT_FLOAT, nargs=2, default=(None, None),
        help='Fit data based on this time interval. (None)')
@add_option('--out_lims', type=OPT_FLOAT, nargs=2, default=(None, None),
        help='Show fitted data over this time interval. (None)')
@add_option('-o', '--save_fig', type=click.Path(), default=None,
        help='Save figure to path. (None)')
@add_option('-x', '--save_xvg', type=click.Path(), default=None,
        help='Save read data to path. (None)')
@add_option('--print/--noprint', default=True,
        help='Print fitting parameters.')
@add_option('--show/--noshow', default=True,
        help='Whether or not to draw graph. (True)')
@add_option('--loglog', is_flag=True,
        help='Scale graph axes logarithmically.')
@add_option('--legend/--nolegend', default=True,
        help='Show a legend.')
@add_option('--xlim', type=OPT_FLOAT, nargs=2, default=(None, None),
        metavar='MIN MAX', help='Set limits on the x axis.')
@add_option('--ylim', type=OPT_FLOAT, nargs=2, default=(None, None),
        metavar='MIN MAX', help='Set limits on the y axis.')
@add_option('--title', default='Droplet spreading with fit',
        help='Figure title.')
@add_option('--xlabel', default='Time (ps)',
        help='Label of x axis.')
@add_option('--ylabel', default='Radius (nm)',
        help='Label of y axis.')
def spreading_fit_cli(files, **kwargs):
    """Fit spreading data of input FILES as power law data.

    Input files must be in whitespace separated format, the first column
    designating time with all following being their corresponding spreading
    radii.

    The combined data can be saved to disk in an XmGrace compatible format.

    """

    do_print = kwargs.pop('print')

    fit_params = fit_spreading_data(files, **kwargs)

    if do_print:
        print('Found fitting parameters (a, k) for r = a*t^k:')
        for a, k in fit_params:
            print('%f %f' % (a, k))




# Combined interface tools for collectiong and plotting
@strata.group()
def interface(name=cmd_interface['name'], short_help=cmd_interface['desc']):
    """Work with interface data of droplets."""
    pass


# Interface collect wrapper
@interface.command(name=cmd_intcollect['name'], short_help=cmd_intcollect['desc'])
@add_argument('base', type=str)
@add_argument('output', type=str)
@add_option('--recenter',
        default='zero', type=click.Choice(['off', 'com', 'zero']),
        help='Center the interface around zero or the center of mass. (zero)')
@add_option('-co', '--cutoff', type=float, default=None,
        help='Boundary bins require this much mass. (None)')
@add_option('-cr', '--cutoff_radius', default=1.,
        help='Boundary bins search for neighbours within this radius. (1 nm)')
@add_option('-cb', '--cutoff_bins', default=1,
        help='Boundary bins require this many neighbours.')
@add_option('--ylim', 'ylims', type=OPT_FLOAT, nargs=2, default=(None, None),
        help='Set limits on the y axis.')
@add_option('-b', '--begin', default=1,
        type=click.IntRange(0, None), metavar='INTEGER',
        help='Begin reading from BASE at this number. (1)')
@add_option('-e', '--end', default=None,
        type=click.IntRange(0, None), metavar='INTEGER',
        help='End reading from BASE at this number. (None)')
@add_option('--ext', default='.dat',
        help='Read using this file extension. (.dat)')
def interface_collect_cli(base, output, **kwargs):
    """Collect the interface boundaries for input files at BASE to OUTPUT.

    The interface is calculated at each height by finding the outermost bins
    fulfilling set criteria to be considered parts of the droplet. For each
    considered bin in the bottom-most layer which has more mass than a set
    cut-off, a search is made for similarly filled bins within a set radius.
    If the number of filled bins within this radius surpasses the final
    requirement, the bin is considered to be connected to the main droplet.
    The left- and rightmost of these bins in the selected layer are taken
    as its boundary cells.

    Read data files must have data fields corresponding to coordinates
    and mass.

    File names are generated by joining the base path and extension with
    a five-digit integer signifying file number ('%s%05d%s').

    """

    set_none_to_inf(kwargs)
    if kwargs['recenter'] == 'off': kwargs['recenter'] = None
    xs, ys = collect_interfaces(base, output, **kwargs)


# Interface viewing wrapper
@interface.command(name=cmd_intview['name'], short_help=cmd_intview['desc'])
@add_argument('base', type=str)
@add_option('-av', '--average', default=1,
        type=click.IntRange(1, None), metavar='INTEGER',
        help='Average interface data in bundles of this size. (1)')
@add_option('-o', '--save_fig', type=str, default=None,
        help='Save figures to base path..')
@add_option('-x', '--save_xvg', type=click.Path(), default='',
        help='Save collected data to disk at base as .xvg file.')
@add_option('--show/--noshow', default=True,
        help='Whether or not to draw graph. (True)')
@add_option('--xlim', type=OPT_FLOAT, nargs=2, default=(None, None),
        metavar='MIN MAX', help='Set limits on the x axis.')
@add_option('--ylim', type=OPT_FLOAT, nargs=2, default=(None, None),
        metavar='MIN MAX', help='Set limits on the y axis.')
@add_option('--title', default='Droplet interface',
        help='Figure title.')
@add_option('--xlabel', default='x (nm)',
        help='Label of x axis.')
@add_option('--ylabel', default='y (nm)',
        help='Label of y axis.')
@add_option('-b', '--begin', default=1,
        type=click.IntRange(0, None), metavar='INTEGER',
        help='Begin reading from BASE at this number. (1)')
@add_option('-e', '--end', default=None,
        type=click.IntRange(0, None), metavar='INTEGER',
        help='End reading from BASE at this number. (None)')
@add_option('--ext', default='.xvg',
        help='Read using this file extension. (.xvg)')
@add_option('--extfig', default='.png',
        help='Save figures using this file extension. (.png)')
def interface_view_cli(base, **kwargs):
    """View the interface boundaries for input files at BASE.

    Optionally averages the interface data over an input bundling length.
    The averaged interfaces can be written to disk as new Grace formatted
    files.

    File names are generated by joining the base path and extension with
    a five-digit integer signifying file number ('%s%05d%s').

    """

    set_none_to_inf(kwargs)
    xs, ys = view_interfaces(base, **kwargs)


# Interface area wrapper
@interface.command(name=cmd_intarea['name'], short_help=cmd_intarea['desc'])
@add_argument('base', type=str)
@add_option('-o', '--save_fig', type=str, default=None,
        help='Save figures to base path..')
@add_option('-x', '--save_xvg', type=click.Path(), default='',
        help='Save collected data to disk at base as .xvg file.')
@add_option('-dt', '--delta_t', default=1.,
        help='Time difference between interface files.')
@add_option('--show/--noshow', default=True,
        help='Whether or not to draw graph. (True)')
@add_option('--xlim', type=OPT_FLOAT, nargs=2, default=(None, None),
        metavar='MIN MAX', help='Set limits on the x axis.')
@add_option('--ylim', type=OPT_FLOAT, nargs=2, default=(None, None),
        metavar='MIN MAX', help='Set limits on the y axis.')
@add_option('--title', default='Droplet interface',
        help='Figure title.')
@add_option('--xlabel', default='t (ps)',
        help='Label of x axis.')
@add_option('--ylabel', default='A (nm^2)',
        help='Label of y axis.')
@add_option('-b', '--begin', default=1,
        type=click.IntRange(0, None), metavar='INTEGER',
        help='Begin reading from BASE at this number. (1)')
@add_option('-e', '--end', default=None,
        type=click.IntRange(0, None), metavar='INTEGER',
        help='End reading from BASE at this number. (None)')
@add_option('--ext', default='.xvg',
        help='Read using this file extension. (.xvg)')
@add_option('--extfig', default='.png',
        help='Save figures using this file extension. (.png)')
def interface_area_cli(base, **kwargs):
    """Calculate the area found within interfaces of input BASE.

    File names are generated by joining the base path and extension with
    a five-digit integer signifying file number ('%s%05d%s').

    """

    set_none_to_inf(kwargs)
    integrate_interfaces(base, **kwargs)


# Interface angle wrapper
@interface.command(name=cmd_intangle['name'], short_help=cmd_intangle['desc'])
@add_argument('base', type=str)
@add_option('--fit/--nofit', default=True,
        help='Measure the contact angle by a circular segment fit. (True)')
@add_option('-h', '--height', type=OPT_FLOAT, default=None,
        help='Measure the contact angle at this height above the substrate. (None)')
@add_option('-dt', '--delta_t', type=float, default=1.,
        help='Time difference between interface files.')
@add_option('-o', '--save_fig', type=str, default=None,
        help='Save figure to path.')
@add_option('-x', '--save_xvg', type=click.Path(), default=None,
        help='Save collected data to path as an .xvg file.')
@add_option('--show/--noshow', default=True,
        help='Whether or not to draw graph. (True)')
@add_option('--xlim', type=OPT_FLOAT, nargs=2, default=(None, None),
        metavar='MIN MAX', help='Set limits on the x axis.')
@add_option('--ylim', type=OPT_FLOAT, nargs=2, default=(None, None),
        metavar='MIN MAX', help='Set limits on the y axis.')
@add_option('--title', default='Contact angles',
        help='Figure title.')
@add_option('--xlabel', default='t (ps)',
        help='Label of x axis.')
@add_option('--ylabel', default='angle (deg.)',
        help='Label of y axis.')
@add_option('-b', '--begin', default=1,
        type=click.IntRange(0, None), metavar='INTEGER',
        help='Begin reading from BASE at this number. (1)')
@add_option('-e', '--end', default=None,
        type=click.IntRange(0, None), metavar='INTEGER',
        help='End reading from BASE at this number. (None)')
@add_option('--ext', default='.xvg',
        help='Read using this file extension. (.xvg)')
def interface_angle_cli(base, **kwargs):
    """Calculate the contact angle for input files at BASE.

    The contact angle can be measured in two ways which may be combined:
    Either by assuming that the interface is well fitted by a circular
    segment and supplying the keyword argument `fig` or by measuring
    the contact angle of both edges by supplying a height with the
    keyword argument `height`.

    The circular segment is not exactly fitted but calculated through the
    chord length and maximum height above the substrate. A measurement
    is made through simple trigonometrics and the mean is taken of both
    edges.

    """

    set_none_to_inf(kwargs)
    interface_contact_angle(base, **kwargs)

# Contact line averaging wrapper
@strata.command(name=cmd_contactline['name'],
        short_help=cmd_contactline['desc'])
@add_argument('base', type=str)
@add_argument('output', type=str)
@add_option('-av', '--average', default=1,
        type=click.IntRange(1, None), metavar='INTEGER',
        help='Sample average the extracted data of this many files.')
@add_option('--rolling/--norolling', default=False,
        help='Perform a rolling average over the data. (False)')
@add_option('--recenter/--norecenter', default=True,
        help='Recenter the extracted edges around zero. (True)')
@add_option('-ea', '--extract_area', type=float, default=(0., 0.), nargs=2,
        help='Extract area of this size. (1 nm, 1 nm)')
@add_option('-eh', '--extract_height', type=float, default=0.,
        help='Synchronise extraction box position at this interface height. (0 nm)')
@add_option('-co', '--cutoff', type=float, default=None,
        help='Boundary bins require this much mass. (0)')
@add_option('-cr', '--cutoff_radius', default=1.,
        help='Boundary bins search for neighbours within this radius. (1 nm)')
@add_option('-cb', '--cutoff_bins', default=1,
        help='Boundary bins require this many neighbours (1).')
@add_option('-b', '--begin', default=1,
        type=click.IntRange(0, None), metavar='INTEGER',
        help='Begin reading from BASE at this number. (1)')
@add_option('-e', '--end', default=None,
        type=click.IntRange(0, None), metavar='INTEGER',
        help='End reading from BASE at this number. (None)')
@add_option('--ext', default='.dat',
        help='Read and write using this file extension. (.dat)')
def cl_average_cli(base, output, **kwargs):
    """Extract the contact line area of files at BASE and write to OUTPUT.

    The contact line area is determined as: For each considered bin
    in the bottom-most layer which has more mass than a set cut-off, a
    search is made for similarly filled bins within a set radius. If the
    number of filled bins within this radius surpasses the final requirement,
    the bin is considered to be connected to the main droplet. The left- and
    rightmost of these bins in the selected layer are taken as the contact
    line edge bins.

    From these bins an area of input extraction size into the bulk is
    included, as well as any cells up to and including the interface
    at each height.

    The data can be sample averaged by supplying a number of files to
    average over. In this case all input files must be of similar coordinate
    grid spacings. Additionally, a rolling average can be performed.

    File names are generated by joining the base path and extension with
    a five-digit integer signifying file number ('%s%05d%s').

    """

    set_none_to_inf(kwargs)
    extract_contact_line_bins(base, output, **kwargs)


@strata.group()
def view(name=cmd_flowview['name'], short_help=cmd_flowview['desc']):
    """Visualise the data of binned map files."""
    pass


@view.command(name=cmd_contourmap['name'], short_help=cmd_contourmap['desc'])
@add_argument('files', type=click.Path(exists=True), nargs=-1)
@add_option('-o', '--save_fig', type=click.Path(), default=None,
        help='Save figure to path. (None)')
@add_option('-l', '--label', type=click.Choice(['M', 'N', 'T', 'U', 'V', 'flow']),
        default='M', help='Label of data to use as height map. (M)')
@add_option('-n', '--num_contours', type=int, default=10,
        help='Number of levels to draw. (10)')
@add_option('-lv', '--levels', 'contour_levels', default=None, type=STR_FLOATS,
        help='Explicit levels to draw contours at.')
@add_option('-c', '--colours', 'contour_colours', default=None, type=STR_LIST,
        help='List of colours to cycle through for drawn contours.')
@add_option('--filled', is_flag=True, default=False,
        help='Fill contour levels.')
@add_option('--vlim', nargs=2, default=(None, None), type=OPT_FLOAT,
        metavar='MIN MAX', help='Set limits for the contour values.')
@add_option('--colourbar/--nocolourbar', 'colorbar', default=False,
        help='Whether or not to draw a colour bar. (False)')
@add_option('-cmap', '--colourmap', 'colormap', default='viridis', type=str,
        help='Set a colour map. (viridis)')
@add_option('--show/--noshow', default=True,
        help='Whether or not to draw graph. (True)')
@add_option('--xlim', type=OPT_FLOAT, nargs=2, default=(None, None),
        metavar='MIN MAX', help='Set limits on the x axis.')
@add_option('--ylim', type=OPT_FLOAT, nargs=2, default=(None, None),
        metavar='MIN MAX', help='Set limits on the y axis.')
@add_option('--title', default='Droplet contours',
        help='Figure title.')
@add_option('--xlabel', default='x (nm)',
        help='Label of x axis.')
@add_option('--ylabel', default='y (nm)',
        help='Label of y axis.')
def view_contour_cli(files, **kwargs):
    """Draw contour maps of the data in FILES.

    The contour levels are by default chosen automatically, but can be
    controlled by setting the number of lines to draw or by supplying limits
    for curves to be drawn within. Additionally the curve levels can be
    supplied explicitly. Note that this option overrides both the input
    limits and number of curves to draw!

    """

    view_flowmap_2d(*files, type='contour', **kwargs)


@view.command(name=cmd_heightmap['name'], short_help=cmd_heightmap['desc'])
@add_argument('files', type=click.Path(exists=True), nargs=-1)
@add_option('-o', '--save_fig', type=click.Path(), default=None,
        help='Save figure to path. (None)')
@add_option('-l', '--label', type=click.Choice(['M', 'N', 'T', 'U', 'V', 'flow']),
        default='M', help='Label of data to use as height map. (M)')
@add_option('--clim', nargs=2, default=(None, None), type=OPT_FLOAT,
        metavar='MIN MAX', help='Set cut-offs for the binned values to include.')
@add_option('--vlim', nargs=2, default=(None, None), type=OPT_FLOAT,
        metavar='MIN MAX', help='Set limits for the shown colour values.')
@add_option('--colourbar/--nocolourbar', 'colorbar', default=True,
        help='Whether or not to draw a colour bar. (True)')
@add_option('-cmap', '--colourmap', 'colormap', default='viridis', type=str,
        help='Set a colour map. (viridis)')
@add_option('--show/--noshow', default=True,
        help='Whether or not to draw graph. (True)')
@add_option('--xlim', type=OPT_FLOAT, nargs=2, default=(None, None),
        metavar='MIN MAX', help='Set limits on the x axis.')
@add_option('--ylim', type=OPT_FLOAT, nargs=2, default=(None, None),
        metavar='MIN MAX', help='Set limits on the y axis.')
@add_option('--title', default='Droplet height map',
        help='Figure title.')
@add_option('--xlabel', default='x (nm)',
        help='Label of x axis.')
@add_option('--ylabel', default='y (nm)',
        help='Label of y axis.')
def view_heightmap_cli(files, **kwargs):
    """Draw 2D height maps of input FILES.

    The height maps can be controlled by supplying limits of values. These
    come in two shapes: One is a cut-off that does not show any bins with
    values outside of the limits (--clim). The other sets limits o the
    colour scale, replacing values outside of these limits with the colour
    representing the corresponding limit (--vlim).

    """

    view_flowmap_2d(*files, type='height', **kwargs)


@view.command(name=cmd_quiver['name'], short_help=cmd_quiver['desc'])
@add_argument('files', type=click.Path(exists=True), nargs=-1)
@add_option('-o', '--save_fig', type=click.Path(), default=None,
        help='Save figure to path. (None)')
@add_option('-co', '--cutoff', type=float, default=None,
        help='Minimum mass of bins to draw fields for. (0)')
@add_option('-cl', '--colour_label', 'colour',
        type=click.Choice(['M', 'N', 'T', 'U', 'V', 'None', 'flow']), default='T',
        help='Colour the flow by values of this label.')
@add_option('--scale', default=1., help='Scale for quiver arrows. (1)')
@add_option('--width', default=0.0015, help='Width of quiver arrows. (0.0015)')
@add_option('--pivot', default='middle', type=click.Choice(['tail', 'middle', 'tip']),
        help='Pivot for flow arrows. (middle)')
@add_option('--vlim', nargs=2, default=(None, None), type=OPT_FLOAT,
        metavar='MIN MAX', help='Set limits for the shown colour values.')
@add_option('--colourbar/--nocolourbar', 'colorbar', default=False,
        help='Whether or not to draw a colour bar. (True)')
@add_option('-cmap', '--colourmap', 'colormap', default='viridis', type=str,
        help='Set a colour map. (viridis)')
@add_option('--show/--noshow', default=True,
        help='Whether or not to draw graph. (True)')
@add_option('--dpi', type=int, default=150,
        help='Set a dpi value for figure output. (150)')
@add_option('--xlim', type=OPT_FLOAT, nargs=2, default=(None, None),
        metavar='MIN MAX', help='Set limits on the x axis.')
@add_option('--ylim', type=OPT_FLOAT, nargs=2, default=(None, None),
        metavar='MIN MAX', help='Set limits on the y axis.')
@add_option('--title', default='Droplet height map',
        help='Figure title.')
@add_option('--xlabel', default='x (nm)',
        help='Label of x axis.')
@add_option('--ylabel', default='y (nm)',
        help='Label of y axis.')
def view_quiver_cli(files, **kwargs):
    """Draw flow fields of input FILES.

    The flow fields can be shown only for certain bins by supplying
    a mass cut-off value (--cutoff). They can be coloured by supplying
    a data label (--colour_label), by default the temperature ('T') is
    shown.

    """

    view_flowfields(*files, cutoff_label='M', **kwargs)


@strata.command(name=cmd_sample['name'], short_help=cmd_sample['desc'])
@add_argument('base', type=str)
@add_argument('label', type=str)
@add_option('-o', '--output', type=click.Path(), default=None,
        help='Write the collected data to disk.')
@add_option('-dt', '--delta_t', 'dt', default=1.,
        help='Time difference between data map files. (1)')
@add_option('--sum/--nosum', default=False,
        help='Whether or not to sample the summed quantity. (False)')
@add_option('-co', '--cutoff', type=float, default=None,
        help='Boundary bins require this much mass. (0)')
@add_option('-cl', '--cutoff_label', type=str, default=None,
        help='Label to use for cutoff. Defaults to input sample label. (None)')
@add_option('-b', '--begin', default=1,
        type=click.IntRange(0, None), metavar='INTEGER',
        help='Begin reading from BASE at this number. (1)')
@add_option('-e', '--end', default=None,
        type=click.IntRange(0, None), metavar='INTEGER',
        help='End reading from BASE at this number. (None)')
@add_option('--ext', default='.dat',
        help='Read using this file extension. (.dat)')
def sample_average_cli(base, label, **kwargs):
    """Sample average data of input label in files of input base."""

    set_none_to_inf(kwargs)
    sample_average_files(base, label, **kwargs)


def set_none_to_inf(kwargs, label='end'):
    if kwargs[label] == None:
        kwargs[label] = np.inf
