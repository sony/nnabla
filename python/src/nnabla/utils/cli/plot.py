# Copyright 2018,2019,2020,2021 Sony Corporation.
# Copyright 2021 Sony Group Corporation.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


def plot_series_command(args):
    import nnabla.monitor as M
    plot_any_command(args, M.plot_series)
    return True


def plot_timer_command(args):
    import nnabla.monitor as M
    format_unit = dict(
        s='seconds',
        m='minutes',
        h='hours',
        d='days',
        )

    if not args.ylabel:
        if args.elapsed:
            args.ylabel = 'Total elapsed time [{}]'.format(
                format_unit[args.time_unit])
        else:
            args.ylabel = 'Elapsed time [{}/iter]'.format(
                format_unit[args.time_unit])
    plot_any_command(args, M.plot_time_elapsed, dict(
        elapsed=args.elapsed, unit=args.time_unit))
    return True


def plot_any_command(args, plot_func, plot_func_kwargs=None):
    if plot_func_kwargs is None:
        plot_func_kwargs = {}
    if args.outfile:
        import matplotlib
        matplotlib.use('Agg')
    if args.label is None:
        args.label = []

    import matplotlib.pyplot as plt
    plt.figure()
    for i, inp in enumerate(args.inputs):
        label = args.label[i] if len(args.label) > i else inp
        plot_func(inp, plot_kwargs=dict(label=label), **plot_func_kwargs)
    plt.legend(loc='best')
    if args.title:
        plt.title(args.title)
    if args.xlabel:
        plt.xlabel(args.xlabel)
    if args.ylabel:
        plt.ylabel(args.ylabel)
    if args.xlim_min is not None:
        plt.xlim(left=args.xlim_min)
    if args.xlim_max is not None:
        plt.xlim(right=args.xlim_max)
    if args.ylim_min is not None:
        plt.ylim(bottom=args.ylim_min)
    if args.ylim_max is not None:
        plt.ylim(top=args.ylim_max)
    if args.outfile:
        plt.savefig(args.outfile)
        return
    plt.show()


def add_plot_any_command(subparsers, subcommand, help, description, plot_command, ylabel_default=''):
    from argparse import RawTextHelpFormatter
    # Train
    subparser = subparsers.add_parser(
        subcommand,
        formatter_class=RawTextHelpFormatter,
        description=description,
        help=help)

    subparser.add_argument('inputs', type=str, nargs='+',
                           metavar='infile', help='Path to input file.')
    subparser.add_argument(
        '-l', '--label', action='append', help='Label of each plot.')
    subparser.add_argument(
        '-o', '--outfile', help='Path to output file.', type=str, default='')
    subparser.add_argument(
        '-x', '--xlabel', help='X-axis label of plot.', default='')
    subparser.add_argument(
        '-y', '--ylabel', help='Y-axis label of plot.', default=ylabel_default)
    subparser.add_argument('-t', '--title', help='Title of plot.', default='')
    subparser.add_argument(
        '-T', '--ylim-max', help='Y-axis plot range max.', type=float, default=None)
    subparser.add_argument(
        '-B', '--ylim-min', help='Y-axis plot range min.', type=float, default=None)
    subparser.add_argument(
        '-R', '--xlim-max', help='X-axis plot range max.', type=float, default=None)
    subparser.add_argument(
        '-L', '--xlim-min', help='X-axis plot range min.', type=float, default=None)
    subparser.set_defaults(func=plot_command)
    return subparser


def add_plot_series_command(subparsers):
    desc = '''Plot *.series.txt files produced by nnabla.MonitorSeries class.

Example:

    nnabla_cli plot_series -x "Epochs" -y "Squared error loss" -T 10 -l "config A" -l "config B" result_a/Training-loss.series.txt result_b/Training-loss.series.txt'''

    add_plot_any_command(
        subparsers, 'plot_series', 'Plot *.series.txt files.', desc, plot_series_command)


def add_plot_timer_command(subparsers):
    desc = '''Plot *.timer.txt files produced by nnabla.MonitorTimeElapsed class.

Example:

    nnabla_cli plot_timer -x "Epochs" -l "config A" -l "config B" result_a/Epoch-time.timer.txt result_b/Epoch-time.timer.txt'''

    subparser = add_plot_any_command(
        subparsers, 'plot_timer', 'Plot *.timer.txt files.', desc, plot_timer_command)
    subparser.add_argument(
        '-e', '--elapsed', help='Plot total elapsed time. By default, it plots elapsed time per iteration.', action='store_true')
    subparser.add_argument(
        '-u', '--time-unit', help='Time unit chosen from {s|m|h|d}.', default='s')
