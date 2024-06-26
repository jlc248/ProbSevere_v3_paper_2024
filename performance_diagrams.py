"""Methods for plotting performance diagram."""
import bootstrap
import numpy
import matplotlib.colors
import matplotlib.pyplot as pyplot

DEFAULT_LINE_COLOUR = numpy.array([228, 26, 28], dtype=float) / 255
DEFAULT_LINE_WIDTH = 2
DEFAULT_BIAS_LINE_COLOUR = numpy.full(3, 152. / 255)
DEFAULT_BIAS_LINE_WIDTH = 1

LEVELS_FOR_CSI_CONTOURS = numpy.linspace(0, 1, num=11, dtype=float)
LEVELS_FOR_BIAS_CONTOURS = numpy.array(
    [0.25, 0.5, 0.75, 1., 1.5, 2., 3., 5.])

BIAS_STRING_FORMAT = '%.2f'
BIAS_LABEL_PADDING_PX = 10

FIGURE_WIDTH_INCHES = 4
FIGURE_HEIGHT_INCHES = 4

FONT_SIZE = 8
pyplot.rc('font', size=FONT_SIZE)
pyplot.rc('axes', titlesize=FONT_SIZE)
pyplot.rc('axes', labelsize=FONT_SIZE)
pyplot.rc('xtick', labelsize=FONT_SIZE)
pyplot.rc('ytick', labelsize=FONT_SIZE)
pyplot.rc('legend', fontsize=FONT_SIZE)
pyplot.rc('figure', titlesize=FONT_SIZE)


def _get_sr_pod_grid(success_ratio_spacing=0.01, pod_spacing=0.01):
    """Creates grid in SR-POD (success ratio / probability of detection) space.
    M = number of rows (unique POD values) in grid
    N = number of columns (unique success ratios) in grid
    :param success_ratio_spacing: Spacing between grid cells in adjacent
        columns.
    :param pod_spacing: Spacing between grid cells in adjacent rows.
    :return: success_ratio_matrix: M-by-N numpy array of success ratios.
        Success ratio increases with column index.
    :return: pod_matrix: M-by-N numpy array of POD values.  POD decreases with
        row index.
    """

    num_success_ratios = 1 + int(numpy.ceil(1. / success_ratio_spacing))
    num_pod_values = 1 + int(numpy.ceil(1. / pod_spacing))

    unique_success_ratios = numpy.linspace(0., 1., num=num_success_ratios)
    unique_pod_values = numpy.linspace(0., 1., num=num_pod_values)[::-1]
    return numpy.meshgrid(unique_success_ratios, unique_pod_values)


def _csi_from_sr_and_pod(success_ratio_array, pod_array):
    """Computes CSI (critical success index) from success ratio and POD.
    POD = probability of detection
    :param success_ratio_array: numpy array (any shape) of success ratios.
    :param pod_array: numpy array (same shape) of POD values.
    :return: csi_array: numpy array (same shape) of CSI values.
    """

    return (success_ratio_array ** -1 + pod_array ** -1 - 1.) ** -1


def _bias_from_sr_and_pod(success_ratio_array, pod_array):
    """Computes frequency bias from success ratio and POD.
    POD = probability of detection
    :param success_ratio_array: numpy array (any shape) of success ratios.
    :param pod_array: numpy array (same shape) of POD values.
    :return: frequency_bias_array: numpy array (same shape) of frequency biases.
    """

    return pod_array / success_ratio_array


def _get_csi_colour_scheme():
    """Returns colour scheme for CSI (critical success index).
    :return: colour_map_object: Colour scheme (instance of
        `matplotlib.colors.ListedColormap`).
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`,
        defining the scale of the colour map.
    """

    this_colour_map_object = pyplot.cm.Blues
    this_colour_norm_object = matplotlib.colors.BoundaryNorm(
        LEVELS_FOR_CSI_CONTOURS, this_colour_map_object.N)

    rgba_matrix = this_colour_map_object(this_colour_norm_object(
        LEVELS_FOR_CSI_CONTOURS))
    colour_list = [
        rgba_matrix[i, ..., :-1] for i in range(rgba_matrix.shape[0])
    ]

    colour_map_object = matplotlib.colors.ListedColormap(colour_list)
    colour_map_object.set_under(numpy.array([1, 1, 1]))
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        LEVELS_FOR_CSI_CONTOURS, colour_map_object.N)

    return colour_map_object, colour_norm_object


def _add_colour_bar(
        axes_object, colour_map_object, values_to_colour, min_colour_value,
        max_colour_value, colour_norm_object=None,
        orientation_string='vertical', extend_min=True, extend_max=True,
        fraction_of_axis_length=1., font_size=FONT_SIZE):
    """Adds colour bar to existing axes.
    :param axes_object: Existing axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param values_to_colour: numpy array of values to colour.
    :param min_colour_value: Minimum value in colour map.
    :param max_colour_value: Max value in colour map.
    :param colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`,
        defining the scale of the colour map.  If `colour_norm_object is None`,
        will assume that scale is linear.
    :param orientation_string: Orientation of colour bar ("vertical" or
        "horizontal").
    :param extend_min: Boolean flag.  If True, the bottom of the colour bar will
        have an arrow.  If False, it will be a flat line, suggesting that lower
        values are not possible.
    :param extend_max: Same but for top of colour bar.
    :param fraction_of_axis_length: Fraction of axis length (y-axis if
        orientation is "vertical", x-axis if orientation is "horizontal")
        occupied by colour bar.
    :param font_size: Font size for labels on colour bar.
    :return: colour_bar_object: Colour bar (instance of
        `matplotlib.pyplot.colorbar`) created by this method.
    """

    if colour_norm_object is None:
        colour_norm_object = matplotlib.colors.Normalize(
            vmin=min_colour_value, vmax=max_colour_value, clip=False)

    scalar_mappable_object = pyplot.cm.ScalarMappable(
        cmap=colour_map_object, norm=colour_norm_object)
    scalar_mappable_object.set_array(values_to_colour)

    if extend_min and extend_max:
        extend_string = 'both'
    elif extend_min:
        extend_string = 'min'
    elif extend_max:
        extend_string = 'max'
    else:
        extend_string = 'neither'

    if orientation_string == 'horizontal':
        padding = 0.075
    else:
        padding = 0.05

    colour_bar_object = pyplot.colorbar(
        ax=axes_object, mappable=scalar_mappable_object,
        orientation=orientation_string, pad=padding, extend=extend_string,
        shrink=fraction_of_axis_length)

    colour_bar_object.ax.tick_params(labelsize=font_size)
    return colour_bar_object


def _get_points_in_perf_diagram(observed_labels, forecast_probabilities, nboots=0):
    """Creates points for performance diagram.
    E = number of examples
    T = number of binarization thresholds
    :param observed_labels: length-E numpy array of class labels (integers in
        0...1).
    :param forecast_probabilities: length-E numpy array with forecast
        probabilities of label = 1.
    :return: pod_by_threshold: length-T numpy array of POD (probability of
        detection) values.
    :return: success_ratio_by_threshold: length-T numpy array of success ratios.
    """

    assert numpy.all(numpy.logical_or(
        observed_labels == 0, observed_labels == 1
    ))

    assert numpy.all(numpy.logical_and(
        forecast_probabilities >= 0, forecast_probabilities <= 1
    ))

    observed_labels = observed_labels.astype(int)
    binarization_thresholds = numpy.linspace(0, 1, num=1001, dtype=float)

    num_thresholds = len(binarization_thresholds)
    pod_by_threshold = numpy.full(num_thresholds, numpy.nan)
    csi_by_threshold = numpy.full(num_thresholds,numpy.nan)
    success_ratio_by_threshold = numpy.full(num_thresholds, numpy.nan)
    binary_accuracy_by_threshold = numpy.full(num_thresholds, numpy.nan)
    binary_freq_bias_by_threshold = numpy.full(num_thresholds, numpy.nan)
    pss_by_threshold = numpy.full(num_thresholds, numpy.nan)
    if(nboots > 0):
      #tmp_thresh = numpy.arange(0,1.01,0.01)
      tmp_thresh = numpy.copy(binarization_thresholds)
      tmp_nthresh = len(tmp_thresh)
      pod_by_threshold_05 = numpy.full(tmp_nthresh, numpy.nan)
      pod_by_threshold_95 = numpy.full(tmp_nthresh, numpy.nan)
      success_ratio_by_threshold_05 = numpy.full(tmp_nthresh, numpy.nan)
      success_ratio_by_threshold_95 = numpy.full(tmp_nthresh, numpy.nan)
      sample_ind = numpy.arange(len(observed_labels),dtype=int)
      boot_ind = bootstrap.bootstrap(sample_ind, bootnum=nboots).astype(int) #boot_ind has shape of nboots by len(observed_labels)     
      all_pods = numpy.full((nboots,tmp_nthresh),numpy.nan)
      all_srs = numpy.full((nboots,tmp_nthresh),numpy.nan)

      for k in range(tmp_nthresh):
        print(k)
        for ii in range(nboots): #each row in boot_ind
          tmp_fcsts = forecast_probabilities[boot_ind[ii]]; tmp_obs = observed_labels[boot_ind[ii]]
          fcst_labs = (tmp_fcsts >= tmp_thresh[k]).astype(int)
          hits = numpy.sum(numpy.logical_and(fcst_labs == 1, tmp_obs == 1))
          misses = numpy.sum(numpy.logical_and(fcst_labs == 0, tmp_obs == 1))
          FAs = numpy.sum(numpy.logical_and(fcst_labs == 1, tmp_obs == 0))
          all_pods[ii,k] = float(hits) / (hits + misses)
          all_srs[ii,k] = float(hits) / (hits + FAs)
      
      pod05 = numpy.percentile(all_pods,5,axis=0) #shape of binarization_thresholds
      pod95 = numpy.percentile(all_pods,95,axis=0) #shape of binarization_thresholds
      sr05 = numpy.percentile(all_srs,5,axis=0) #shape of binarization_thresholds
      sr95 = numpy.percentile(all_srs,95,axis=0) #shape of binarization_thresholds

    for k in range(num_thresholds):
        these_forecast_labels = (
            forecast_probabilities >= binarization_thresholds[k]
        ).astype(int)

        this_num_hits = numpy.sum(numpy.logical_and(
            these_forecast_labels == 1, observed_labels == 1
        ))

        this_num_false_alarms = numpy.sum(numpy.logical_and(
            these_forecast_labels == 1, observed_labels == 0
        ))

        this_num_misses = numpy.sum(numpy.logical_and(
            these_forecast_labels == 0, observed_labels == 1
        ))

        this_num_cns = numpy.sum(numpy.logical_and(
            these_forecast_labels == 0, observed_labels == 0
        ))

        try:
            pod_by_threshold[k] = (
                float(this_num_hits) / (this_num_hits + this_num_misses)
            )
        except ZeroDivisionError:
            pass
        
        try:
            success_ratio_by_threshold[k] = (
                float(this_num_hits) / (this_num_hits + this_num_false_alarms)
            )
        except ZeroDivisionError:
            pass

        try:
            csi_by_threshold[k] = (
                float(this_num_hits) / (this_num_hits + this_num_misses + this_num_false_alarms)
            )
        except ZeroDivisionError:
            pass

        try:
            binary_accuracy_by_threshold[k] = (
                (float(this_num_hits) + this_num_cns) / (this_num_hits + this_num_misses + this_num_false_alarms + this_num_cns)
            )
        except ZeroDivisionError:
            pass

        try:
            binary_freq_bias_by_threshold[k] = (
                (float(this_num_hits) + this_num_false_alarms) / (this_num_hits + this_num_misses)
            )
        except ZeroDivisionError:
            pass

        try:
            pss_by_threshold[k] = ( 
               ((this_num_hits * this_num_cns) - (this_num_false_alarms * this_num_misses)) / ((this_num_hits + this_num_misses) * (this_num_false_alarms + this_num_cns)) 
            )
        except ZeroDivisionError:
            pass

    pod_by_threshold = numpy.array([1.] + pod_by_threshold.tolist() + [0.])
    success_ratio_by_threshold = numpy.array(
        [0.] + success_ratio_by_threshold.tolist() + [1.]
    )

    if(nboots > 0):
      sr95 = numpy.array([0.] + sr95.tolist() + [1.])
      sr05 = numpy.array([0.] + sr05.tolist() + [1.])
      pod95 = numpy.array([1.] + pod95.tolist() + [0.])
      pod05 = numpy.array([1.] + pod05.tolist() + [0.])
      return {'pod':pod_by_threshold, 'sr':success_ratio_by_threshold, 'csi':csi_by_threshold, \
              'bin_acc':binary_accuracy_by_threshold, 'bin_bias':binary_freq_bias_by_threshold, 'pss':pss_by_threshold, \
              'pod05':pod05, 'pod95':pod95, \
              'sr05':sr05, 'sr95':sr95}
    else:
      return {'pod':pod_by_threshold, 'sr':success_ratio_by_threshold, 'csi':csi_by_threshold, \
              'bin_acc':binary_accuracy_by_threshold, 'bin_bias':binary_freq_bias_by_threshold, 'pss':pss_by_threshold}


def plot_performance_diagram(
        observed_labels, forecast_probabilities,
        line_colour=DEFAULT_LINE_COLOUR, line_width=DEFAULT_LINE_WIDTH,
        bias_line_colour=DEFAULT_BIAS_LINE_COLOUR,
        bias_line_width=DEFAULT_BIAS_LINE_WIDTH,
        nboots=0, return_axes=False):
    """Plots performance diagram.
    E = number of examples
    :param observed_labels: length-E numpy array of class labels (integers in
        0...1).
    :param forecast_probabilities: length-E numpy array with forecast
        probabilities of label = 1.
    :param line_colour: Colour (in any format accepted by `matplotlib.colors`).
    :param line_width: Line width (real positive number).
    :param bias_line_colour: Colour of contour lines for frequency bias.
    :param bias_line_width: Width of contour lines for frequency bias.
    :param nboots: number of bootstaps for CI (default=0).
    :return: pod_by_threshold: See doc for `_get_points_in_perf_diagram`.
        detection) values.
    :return: success_ratio_by_threshold: Same.
    """

    scores_dict = _get_points_in_perf_diagram(
        observed_labels=observed_labels,
        forecast_probabilities=forecast_probabilities,nboots=nboots)
    pod_by_threshold = scores_dict['pod']
    success_ratio_by_threshold = scores_dict['sr']
    

    _, axes_object = pyplot.subplots(
        1, 1, figsize=(FIGURE_WIDTH_INCHES, FIGURE_HEIGHT_INCHES)
    )

    success_ratio_matrix, pod_matrix = _get_sr_pod_grid()
    csi_matrix = _csi_from_sr_and_pod(success_ratio_matrix, pod_matrix)
    frequency_bias_matrix = _bias_from_sr_and_pod(
        success_ratio_matrix, pod_matrix)
    
    this_colour_map_object, this_colour_norm_object = _get_csi_colour_scheme()

    pyplot.contourf(
        success_ratio_matrix, pod_matrix, csi_matrix, LEVELS_FOR_CSI_CONTOURS,
        cmap=this_colour_map_object, norm=this_colour_norm_object, vmin=0.,
        vmax=1., axes=axes_object)

    colour_bar_object = _add_colour_bar(
        axes_object=axes_object, colour_map_object=this_colour_map_object,
        colour_norm_object=this_colour_norm_object,
        values_to_colour=csi_matrix, min_colour_value=0.,
        max_colour_value=1., orientation_string='vertical',
        extend_min=False, extend_max=False)
    colour_bar_object.set_label('CSI (critical success index)')

    bias_colour_tuple = ()
    for _ in range(len(LEVELS_FOR_BIAS_CONTOURS)):
        bias_colour_tuple += (bias_line_colour,)

    bias_contour_object = pyplot.contour(
        success_ratio_matrix, pod_matrix, frequency_bias_matrix,
        LEVELS_FOR_BIAS_CONTOURS, colors=bias_colour_tuple,
        linewidths=bias_line_width, linestyles='dashed', axes=axes_object)
    pyplot.clabel(
        bias_contour_object, inline=True, inline_spacing=BIAS_LABEL_PADDING_PX,
        fmt=BIAS_STRING_FORMAT, fontsize=FONT_SIZE)

    nan_flags = numpy.logical_or(
        numpy.isnan(success_ratio_by_threshold), numpy.isnan(pod_by_threshold)
    )

    if not numpy.all(nan_flags):
        real_indices = numpy.where(numpy.invert(nan_flags))[0]
        line_obj, = axes_object.plot(
            success_ratio_by_threshold[real_indices],
            pod_by_threshold[real_indices], color=line_colour,
            linestyle='solid', linewidth=line_width)

        if(nboots > 0):
          tmp_sr_by_thresh = numpy.copy(success_ratio_by_threshold)
          #print(len(tmp_sr_by_thresh),len(scores_dict['pod05']),len(scores_dict['pod95']))
          #tmp_sr_by_thresh = success_ratio_by_threshold[0::10]  #only works if tmp_thresh = np.arange(0,1.01,0.01) and thresh=np.linspace(0,1,num=1001)
          axes_object.fill_between(tmp_sr_by_thresh, scores_dict['pod05'], scores_dict['pod95'], facecolor=line_colour, alpha=0.3)
 

        xs = [success_ratio_by_threshold[50]] + list(success_ratio_by_threshold[100:1000:100]) + [success_ratio_by_threshold[950]]
        ys = [pod_by_threshold[50]] + list(pod_by_threshold[100:1000:100]) + [pod_by_threshold[950]]
        labs = ['5','10','20','30','40','50','60','70','80','90','95']
        axes_object.plot(xs, ys, color=line_colour, marker='o', linestyle='None', markersize=6)

        for i in range(len(xs)):
          if(i==0):
            xcor = 0.0075; ycor = 0.0075
          else:
            xcor = 0.0125; ycor = 0.0075
          axes_object.annotate(labs[i], xy=(xs[i]-xcor, ys[i]-ycor), color='white', fontsize=4)

    axes_object.set_xlabel('Success ratio (1 - FAR)')
    axes_object.set_ylabel('POD (probability of detection)')
    axes_object.set_xlim(0., 1.)
    axes_object.set_ylim(0., 1.)

    if(return_axes):
      return scores_dict, axes_object, line_obj
    else:
      return scores_dict
