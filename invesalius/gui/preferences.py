import wx.lib.colourselect as csel
import invesalius.gui.widgets.gradient as grad
import sys
import os

from functools import partial
import nibabel as nb
import numpy as np
import matplotlib.pyplot as plt

import invesalius.constants as const
import invesalius.session as ses
import invesalius.gui.dialogs as dlg
import invesalius.data.vtk_utils as vtk_utils
from invesalius import inv_paths

import wx
from invesalius import utils
from invesalius.gui.language_dialog import ComboBoxLanguage
from invesalius.net.pedal_connection import PedalConnector
from invesalius.pubsub import pub as Publisher
from invesalius.i18n import tr as _

from invesalius.navigation.tracker import Tracker
from invesalius.navigation.robot import Robot
from invesalius.net.neuronavigation_api import NeuronavigationApi
from invesalius.navigation.navigation import Navigation


class Preferences(wx.Dialog):
    def __init__(
        self,
        parent,
        page,
        id_=-1,
        title=_("Preferences"),
        style=wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER
    ):
        super().__init__(parent, id_, title, style=style)

        self.book = wx.Notebook(self, -1)

        self.visualization_tab = VisualizationTab(self.book)
        self.language_tab = LanguageTab(self.book)

        self.book.AddPage(self.visualization_tab, _("Visualization"))

        session = ses.Session()
        mode = session.GetConfig('mode')
        if mode == const.MODE_NAVIGATOR:
            tracker = Tracker()
            robot = Robot()
            neuronavigation_api = NeuronavigationApi()
            pedal_connector = PedalConnector(neuronavigation_api, self)
            navigation = Navigation(
                pedal_connector=pedal_connector,
                neuronavigation_api=neuronavigation_api,
            )
            self.navigation_tab = NavigationTab(self.book, navigation)
            self.tracker_tab = TrackerTab(self.book, tracker, robot)
            self.object_tab = ObjectTab(
                self.book, navigation, tracker, pedal_connector, neuronavigation_api)

            self.book.AddPage(self.navigation_tab, _("Navigation"))
            self.book.AddPage(self.tracker_tab, _("Tracker"))
            self.book.AddPage(self.object_tab, _("Stimulator"))

        self.book.AddPage(self.language_tab, _("Language"))

        btnsizer = self.CreateStdDialogButtonSizer(wx.OK | wx.CANCEL)
        min_width = max([i.GetMinWidth() for i in (self.book.GetChildren())])
        min_height = max([i.GetMinHeight() for i in (self.book.GetChildren())])
        if sys.platform.startswith("linux"):
            self.book.SetMinClientSize((min_width * 2, min_height * 2))
        self.book.SetSelection(page)

        # Bind OK
        self.Bind(wx.EVT_BUTTON, self.OnOK, id=wx.ID_OK)

        sizer = wx.BoxSizer(wx.VERTICAL)
        sizer.Add(self.book, 1, wx.EXPAND | wx.ALL)
        sizer.Add(btnsizer, 0, wx.GROW | wx.RIGHT | wx.TOP | wx.BOTTOM, 5)
        self.SetSizerAndFit(sizer)
        self.Layout()
        self.__bind_events()

    def __bind_events(self):
        Publisher.subscribe(self.LoadPreferences, "Load Preferences")

    def OnOK(self, event):
        Publisher.sendMessage("Save Preferences")
        self.EndModal(wx.ID_OK)
        

    def GetPreferences(self):
        values = {}
        lang = self.language_tab.GetSelection()
        viewer = self.visualization_tab.GetSelection()
        values.update(lang)
        values.update(viewer)

        return values

    def LoadPreferences(self):
        session = ses.Session()

        rendering = session.GetConfig('rendering')
        surface_interpolation = session.GetConfig('surface_interpolation')
        language = session.GetConfig('language')
        slice_interpolation = session.GetConfig('slice_interpolation')
        mode = session.GetConfig('mode')

        if mode == const.MODE_NAVIGATOR:
            self.object_tab.LoadConfig()

        values = {
            const.RENDERING: rendering,
            const.SURFACE_INTERPOLATION: surface_interpolation,
            const.LANGUAGE: language,
            const.SLICE_INTERPOLATION: slice_interpolation,
        }

        self.visualization_tab.LoadSelection(values)
        self.language_tab.LoadSelection(values)


class VisualizationTab(wx.Panel):
    def __init__(self, parent):

        wx.Panel.__init__(self, parent)
        self.__bind_events()

        self.session = ses.Session()

        self.colormaps = ["autumn", "hot", "plasma", "cividis",  # sequential
                          "bwr", "RdBu",  # diverging
                          "Set3", "tab10",  # categorical
                          "twilight", "hsv"]   # cyclic
        self.number_colors = 4
        self.cluster_volume = None

        self.conf = dict(self.session.GetConfig('mep_configuration'))
        self.conf['mep_colormap'] = "autumn"

        bsizer = wx.StaticBoxSizer(wx.VERTICAL, self, _("3D Visualization"))
        lbl_inter = wx.StaticText(
            bsizer.GetStaticBox(), -1, _("Surface Interpolation "))
        rb_inter = self.rb_inter = wx.RadioBox(
            bsizer.GetStaticBox(),
            -1,
            "",
            choices=["Flat", "Gouraud", "Phong"],
            majorDimension=3,
            style=wx.RA_SPECIFY_COLS | wx.NO_BORDER,
        )

        bsizer.Add(lbl_inter, 0, wx.TOP | wx.LEFT | wx.FIXED_MINSIZE, 10)
        bsizer.Add(rb_inter, 0, wx.TOP | wx.LEFT | wx.FIXED_MINSIZE, 0)

        lbl_rendering = wx.StaticText(
            bsizer.GetStaticBox(), -1, _("Volume Rendering"))
        rb_rendering = self.rb_rendering = wx.RadioBox(
            bsizer.GetStaticBox(),
            -1,
            choices=["CPU", _(u"GPU (NVidia video cards only)")],
            majorDimension=2,
            style=wx.RA_SPECIFY_COLS | wx.NO_BORDER,
        )
        bsizer.Add(lbl_rendering, 0, wx.TOP | wx.LEFT | wx.FIXED_MINSIZE, 10)
        bsizer.Add(rb_rendering, 0, wx.TOP | wx.LEFT | wx.FIXED_MINSIZE, 0)

        bsizer_slices = wx.StaticBoxSizer(
            wx.VERTICAL, self, _("2D Visualization"))
        lbl_inter_sl = wx.StaticText(
            bsizer_slices.GetStaticBox(), -1, _("Slice Interpolation "))
        rb_inter_sl = self.rb_inter_sl = wx.RadioBox(
            bsizer_slices.GetStaticBox(),
            -1,
            choices=[_("Yes"), _("No")],
            majorDimension=3,
            style=wx.RA_SPECIFY_COLS | wx.NO_BORDER,
        )
        bsizer_slices.Add(lbl_inter_sl, 0, wx.TOP |
                          wx.LEFT | wx.FIXED_MINSIZE, 10)
        bsizer_slices.Add(rb_inter_sl, 0, wx.TOP |
                          wx.LEFT | wx.FIXED_MINSIZE, 0)

        border = wx.BoxSizer(wx.VERTICAL)
        border.Add(bsizer_slices, 0, wx.EXPAND | wx.ALL | wx.FIXED_MINSIZE, 10)
        border.Add(bsizer, 1, wx.EXPAND | wx.ALL | wx.FIXED_MINSIZE, 10)

        # Creating MEP Mapping BoxSizer
        if self.conf.get('enabled_once') is True:
            self.bsizer_mep = self.InitMEPMapping(None)
            border.Add(self.bsizer_mep, 0, wx.EXPAND |
                       wx.ALL | wx.FIXED_MINSIZE, 10)

        self.SetSizerAndFit(border)
        self.Layout()

    def __bind_events(self):
        Publisher.subscribe(self.InsertNewSurface,
                            'Update surface info in GUI')
        Publisher.subscribe(self.ChangeSurfaceName,
                            'Change surface name')
        Publisher.subscribe(self.OnCloseProject, 'Close project data')
        Publisher.subscribe(self.OnRemoveSurfaces, 'Remove surfaces')

    def GetSelection(self):

        options = {
            const.RENDERING: self.rb_rendering.GetSelection(),
            const.SURFACE_INTERPOLATION: self.rb_inter.GetSelection(),
            const.SLICE_INTERPOLATION: self.rb_inter_sl.GetSelection()
        }

        return options

    def InitMEPMapping(self, event):
        # Adding a new sized for MEP Mapping options
        # Structured as follows:
        # MEP Mapping
        # - Surface Selection -> ComboBox
        # - Gaussian Radius -> SpinCtrlDouble
        # - Gaussian Standard Deviation -> SpinCtrlDouble
        # - Select Colormap -> ComboBox + Image frame
        # - Color Map Values
        # -- Min Value -> SpinCtrlDouble
        # -- Low Value -> SpinCtrlDouble
        # -- Mid Value -> SpinCtrlDouble
        # -- Max Value -> SpinCtrlDouble
        # TODO: Add a button to apply the colormap to the current volume
        # TODO: Store MEP visualization settings in a

        bsizer_mep = wx.StaticBoxSizer(
            wx.VERTICAL, self, _("TMS Motor Mapping"))

        # Surface Selection
        try:
            default_colour = wx.SystemSettings.GetColour(wx.SYS_COLOUR_MENUBAR)
        except AttributeError:
            default_colour = wx.SystemSettings_GetColour(wx.SYS_COLOUR_MENUBAR)
        self.SetBackgroundColour(default_colour)

        self.surface_list = []
        # Combo related to mask name
        combo_surface_name = wx.ComboBox(bsizer_mep.GetStaticBox(), -1,
                                         style=wx.CB_DROPDOWN | wx.CB_READONLY | wx.ALL | wx.EXPAND | wx.GROW)
        # combo_surface_name.SetSelection(0)
        if sys.platform != 'win32':
            combo_surface_name.SetWindowVariant(wx.WINDOW_VARIANT_SMALL)
        combo_surface_name.Bind(wx.EVT_COMBOBOX, self.OnComboName)
        self.combo_surface_name = combo_surface_name

        # Mask colour
        button_colour = csel.ColourSelect(
            bsizer_mep.GetStaticBox(), -1, colour=(0, 0, 255), size=(22, -1))
        button_colour.Bind(csel.EVT_COLOURSELECT, self.OnSelectColour)
        self.button_colour = button_colour

        # Sizer which represents the first line
        line1 = wx.BoxSizer(wx.HORIZONTAL)
        line1.Add(combo_surface_name, 1,  wx.ALL | wx.EXPAND | wx.GROW, 7)
        line1.Add(button_colour, 0,  wx.ALL | wx.EXPAND | wx.GROW, 7)

        surface_sel_lbl = wx.StaticText(
            bsizer_mep.GetStaticBox(), -1, _("Brain Surface:"))
        surface_sel_sizer = wx.BoxSizer(wx.HORIZONTAL)

        surface_sel_sizer.Add(surface_sel_lbl, 0, wx.GROW |
                              wx.EXPAND | wx.LEFT | wx.RIGHT, 5)
        # fixed_sizer.AddSpacer(7)
        surface_sel_sizer.Add(line1, 0, wx.EXPAND |
                              wx.GROW | wx.LEFT | wx.RIGHT, 5)

        # Gaussian Radius Line
        lbl_gaussian_radius = wx.StaticText(
            bsizer_mep.GetStaticBox(), -1, _("Gaussian Radius:"))
        self.spin_gaussian_radius = wx.SpinCtrlDouble(
            bsizer_mep.GetStaticBox(), -1, "", size=wx.Size(64, 23), inc=0.5)
        self.spin_gaussian_radius.Enable(1)
        self.spin_gaussian_radius.SetRange(1, 99)
        self.spin_gaussian_radius.SetValue(self.conf.get("gaussian_radius"))

        self.spin_gaussian_radius.Bind(wx.EVT_TEXT, partial(
            self.OnSelectGaussianRadius, ctrl=self.spin_gaussian_radius))
        self.spin_gaussian_radius.Bind(wx.EVT_SPINCTRL, partial(
            self.OnSelectGaussianRadius, ctrl=self.spin_gaussian_radius))

        line_gaussian_radius = wx.BoxSizer(
            wx.HORIZONTAL)
        line_gaussian_radius.AddMany([
            (lbl_gaussian_radius, 1, wx.EXPAND |
             wx.GROW | wx.TOP | wx.RIGHT | wx.LEFT, 5),
            (self.spin_gaussian_radius, 0, wx.ALL | wx.EXPAND | wx.GROW, 5)
        ])

        # Gaussian Standard Deviation Line
        lbl_std_dev = wx.StaticText(
            bsizer_mep.GetStaticBox(), -1, _("Gaussian Standard Deviation:"))
        self.spin_std_dev = wx.SpinCtrlDouble(
            bsizer_mep.GetStaticBox(), -1, "", size=wx.Size(64, 23), inc=0.01)
        self.spin_std_dev.Enable(1)
        self.spin_std_dev.SetRange(0.01, 5.0)
        self.spin_std_dev.SetValue(self.conf.get("gaussian_sharpness"))

        self.spin_std_dev.Bind(wx.EVT_TEXT, partial(
            self.OnSelectStdDev, ctrl=self.spin_std_dev))
        self.spin_std_dev.Bind(wx.EVT_SPINCTRL, partial(
            self.OnSelectStdDev, ctrl=self.spin_std_dev))

        line_std_dev = wx.BoxSizer(wx.HORIZONTAL)
        line_std_dev.AddMany([
            (lbl_std_dev, 1, wx.EXPAND |
             wx.GROW | wx.TOP | wx.RIGHT | wx.LEFT, 5),
            (self.spin_std_dev, 0, wx.ALL | wx.EXPAND | wx.GROW, 5)
        ])

        # Select Colormap Line
        lbl_colormap = wx.StaticText(
            bsizer_mep.GetStaticBox(), -1, _("Select Colormap:"))

        self.combo_thresh = wx.ComboBox(bsizer_mep.GetStaticBox(), -1, "",  # size=(15,-1),
                                        choices=self.colormaps,
                                        style=wx.CB_DROPDOWN | wx.CB_READONLY)
        self.combo_thresh.Bind(wx.EVT_COMBOBOX, self.OnSelectColormap)
        # by default use the initial value set in the configuration
        self.combo_thresh.SetSelection(
            self.colormaps.index(self.conf.get('mep_colormap')))

        cmap = plt.get_cmap(self.conf.get('mep_colormap'))
        colors_gradient = self.GenerateColormapColors(cmap)

        self.gradient = grad.GradientDisp(bsizer_mep.GetStaticBox(), -1, -5000, 5000, -5000, 5000,
                                          colors_gradient)

        colormap_gradient_sizer = wx.BoxSizer(
            wx.HORIZONTAL)
        colormap_gradient_sizer.AddMany(
            [
                (self.combo_thresh, 0, wx.EXPAND |
                    wx.GROW | wx.LEFT | wx.RIGHT, 5),
                (self.gradient, 1, wx.EXPAND | wx.LEFT | wx.RIGHT, 5)]
        )

        colormap_sizer = wx.BoxSizer(
            wx.VERTICAL)
        spacer = wx.StaticText(bsizer_mep.GetStaticBox(), -1, "")

        colormap_sizer.AddMany(
            [
                (lbl_colormap, 0, wx.GROW | wx.EXPAND | wx.LEFT | wx.RIGHT, 5),
                (spacer, 0, wx.GROW | wx.EXPAND | wx.LEFT | wx.RIGHT, 5),
                (colormap_gradient_sizer, 0, wx.GROW | wx.SHRINK | wx.LEFT | wx.RIGHT, 5)]
        )

        colormap_custom = wx.BoxSizer(
            wx.VERTICAL)

        lbl_colormap_ranges = wx.StaticText(
            bsizer_mep.GetStaticBox(), -1, _("Custom Colormap Ranges"))
        lbl_colormap_ranges.SetFont(wx.Font(9, wx.DEFAULT, wx.NORMAL, wx.BOLD))

        lbl_min = wx.StaticText(
            bsizer_mep.GetStaticBox(), -1, _("Min Value (uV):"))

        self.spin_min = wx.SpinCtrlDouble(
            bsizer_mep.GetStaticBox(), -1, "", size=wx.Size(70, 23), inc=10)
        self.spin_min.Enable(1)
        self.spin_min.SetRange(0, 10000)
        self.spin_min.SetValue(self.conf.get('colormap_range_uv').get('min'))
        line_cm_min = wx.BoxSizer(wx.HORIZONTAL)
        line_cm_min.AddMany([
            (lbl_min, 1, wx.EXPAND | wx.GROW | wx.TOP | wx.RIGHT | wx.LEFT, 5),
            (self.spin_min, 0, wx.ALL | wx.EXPAND | wx.GROW, 5)
        ])

        lbl_low = wx.StaticText(
            bsizer_mep.GetStaticBox(), -1, _("Low Value (uV):"))
        self.spin_low = wx.SpinCtrlDouble(
            bsizer_mep.GetStaticBox(), -1, "", size=wx.Size(70, 23), inc=10)
        self.spin_low.Enable(1)
        self.spin_low.SetRange(0, 10000)
        self.spin_low.SetValue(self.conf.get('colormap_range_uv').get('low'))
        line_cm_low = wx.BoxSizer(wx.HORIZONTAL)
        line_cm_low.AddMany([
            (lbl_low, 1, wx.EXPAND | wx.GROW | wx.TOP | wx.RIGHT | wx.LEFT, 5),
            (self.spin_low, 0, wx.ALL | wx.EXPAND | wx.GROW, 5)
        ])

        lbl_mid = wx.StaticText(
            bsizer_mep.GetStaticBox(), -1, _("Mid Value (uV):"))
        self.spin_mid = wx.SpinCtrlDouble(
            bsizer_mep.GetStaticBox(), -1, "", size=wx.Size(70, 23), inc=10)
        self.spin_mid.Enable(1)
        self.spin_mid.SetRange(0, 10000)
        self.spin_mid.SetValue(self.conf.get('colormap_range_uv').get('mid'))
        line_cm_mid = wx.BoxSizer(wx.HORIZONTAL)
        line_cm_mid.AddMany([
            (lbl_mid, 1, wx.EXPAND | wx.GROW | wx.TOP | wx.RIGHT | wx.LEFT, 5),
            (self.spin_mid, 0, wx.ALL | wx.EXPAND | wx.GROW, 5)
        ])

        lbl_max = wx.StaticText(
            bsizer_mep.GetStaticBox(), -1, _("Max Value (uV):"))
        self.spin_max = wx.SpinCtrlDouble(
            bsizer_mep.GetStaticBox(), -1, "", size=wx.Size(70, 23), inc=10)
        self.spin_max.Enable(1)
        self.spin_max.SetRange(0, 10000)
        self.spin_max.SetValue(self.conf.get('colormap_range_uv').get('max'))
        line_cm_max = wx.BoxSizer(wx.HORIZONTAL)
        line_cm_max.AddMany([
            (lbl_max, 1, wx.EXPAND | wx.GROW | wx.TOP | wx.RIGHT | wx.LEFT, 5),
            (self.spin_max, 0, wx.ALL | wx.EXPAND | wx.GROW, 5)
        ])

        # Binding events for the colormap ranges
        for ctrl in zip([self.spin_min, self.spin_low, self.spin_mid, self.spin_max], ['min', 'low', 'mid', 'max']):
            ctrl[0].Bind(wx.EVT_TEXT, partial(
                self.OnSelectColormapRange, ctrl=ctrl[0], key=ctrl[1]))
            ctrl[0].Bind(wx.EVT_SPINCTRL, partial(
                self.OnSelectColormapRange, ctrl=ctrl[0], key=ctrl[1]))

        colormap_custom.AddMany([
            (lbl_colormap_ranges, 0, wx.TOP | wx.LEFT, 10),
            (line_cm_min, 0, wx.GROW | wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 5),
            (line_cm_low, 0, wx.GROW | wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 5),
            (line_cm_mid, 0, wx.GROW | wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 5),
            (line_cm_max, 0, wx.GROW | wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 5)
        ])

        # Reset to defaults button
        btn_reset = wx.Button(
            bsizer_mep.GetStaticBox(), -1, _("Reset to defaults"))
        btn_reset.Bind(wx.EVT_BUTTON, self.ResetMEPSettings)

        # centered button reset
        colormap_custom.Add(btn_reset, 0, wx.ALIGN_CENTER | wx.TOP, 10)

        colormap_sizer.Add(colormap_custom, 0, wx.GROW |
                           wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 5)
        bsizer_mep.AddMany([
            (surface_sel_sizer, 0, wx.GROW | wx.EXPAND |
             wx.LEFT | wx.RIGHT | wx.TOP, 5),
            (line_gaussian_radius, 0, wx.GROW | wx.EXPAND |
             wx.LEFT | wx.RIGHT | wx.TOP, 5),
            (line_std_dev, 0, wx.GROW | wx.EXPAND |
             wx.LEFT | wx.RIGHT | wx.TOP, 5),
            (colormap_sizer, 0, wx.GROW | wx.EXPAND |
                wx.LEFT | wx.RIGHT | wx.TOP, 5)]
        )

        return bsizer_mep

    def ResetMEPSettings(self, event):
        # fire an event that will reset the MEP settings to the default values in MEP Visualizer
        Publisher.sendMessage('Reset MEP Config')
        # self.session.SetConfig('mep_configuration', self.conf)
        self.UpdateMEPFromSession()

    def UpdateMEPFromSession(self):
        self.conf = dict(self.session.GetConfig('mep_configuration'))
        self.spin_gaussian_radius.SetValue(self.conf.get('gaussian_radius'))
        self.spin_std_dev.SetValue(self.conf.get('gaussian_sharpness'))

        self.combo_thresh.SetSelection(
            self.colormaps.index(self.conf.get('mep_colormap')))
        partial(self.OnSelectColormap, event=None, ctrl=self.combo_thresh)
        partial(self.OnSelectColormapRange, event=None,
                ctrl=self.spin_min, key='min')

        ranges = self.conf.get('colormap_range_uv')
        ranges = dict(ranges)
        self.spin_min.SetValue(ranges.get('min'))
        self.spin_low.SetValue(ranges.get('low'))
        self.spin_mid.SetValue(ranges.get('mid'))
        self.spin_max.SetValue(ranges.get('max'))

    def OnSelectStdDev(self, evt, ctrl):
        self.conf['gaussian_sharpness'] = ctrl.GetValue()
        # Save the configuration
        self.session.SetConfig('mep_configuration', self.conf)

    def OnSelectGaussianRadius(self, evt, ctrl):
        self.conf['gaussian_radius'] = ctrl.GetValue()
        # Save the configuration
        self.session.SetConfig('mep_configuration', self.conf)

    def OnSelectColormapRange(self, evt, ctrl, key):
        self.conf['colormap_range_uv'][key] = ctrl.GetValue()
        self.session.SetConfig('mep_configuration', self.conf)

    def LoadSelection(self, values):
        rendering = values[const.RENDERING]
        surface_interpolation = values[const.SURFACE_INTERPOLATION]
        slice_interpolation = values[const.SLICE_INTERPOLATION]

        self.rb_rendering.SetSelection(int(rendering))
        self.rb_inter.SetSelection(int(surface_interpolation))
        self.rb_inter_sl.SetSelection(int(slice_interpolation))

    def OnSelectColormap(self, event=None):
        self.conf['mep_colormap'] = self.colormaps[self.combo_thresh.GetSelection(
        )]
        colors = self.GenerateColormapColors(
            self.conf.get('mep_colormap'), self.number_colors)

        # Save the configuration
        self.session.SetConfig('mep_configuration', self.conf)

        self.UpdateGradient(self.gradient, colors)

        if isinstance(self.cluster_volume, np.ndarray):
            self.apply_colormap(self.conf.get('mep_colormap'),
                                self.cluster_volume, self.zero_value)

    def GenerateColormapColors(self, colormap_name, number_colors=4):
        cmap = plt.get_cmap(colormap_name)
        colors_gradient = [(int(255*cmap(i)[0]),
                            int(255*cmap(i)[1]),
                            int(255*cmap(i)[2]),
                            int(255*cmap(i)[3])) for i in np.linspace(0, 1, number_colors)]

        return colors_gradient

    def UpdateGradient(self, gradient, colors):
        gradient.SetGradientColours(colors)
        gradient.Refresh()
        gradient.Update()

        self.Refresh()
        self.Update()
        self.Show(True)

    def apply_colormap(self, colormap, cluster_volume, zero_value):
        # 2. Attribute different hue accordingly
        cmap = plt.get_cmap(colormap)

        # new way
        # Flatten the data to 1D
        cluster_volume_unique = np.unique(cluster_volume)
        # Map the scaled data to colors
        colors = cmap(cluster_volume_unique / 255)
        # Create a dictionary where keys are scaled data and values are colors
        color_dict = {val: color for val, color in zip(
            cluster_volume_unique, map(tuple, colors))}

        self.slc.aux_matrices_colours['color_overlay'] = color_dict
        # add transparent color for nans and non GM voxels
        if zero_value in self.slc.aux_matrices_colours['color_overlay']:
            self.slc.aux_matrices_colours['color_overlay'][zero_value] = (
                0.0, 0.0, 0.0, 0.0)
        else:
            print("Zero value not found in color_overlay. No data is set as transparent.")

        Publisher.sendMessage('Reload actual slice')

    def OnRemoveSurfaces(self, surface_indexes):
        s = self.combo_surface_name.GetSelection()
        ns = 0

        old_dict = self.surface_list
        new_dict = []
        i = 0
        for n, (name, index) in enumerate(old_dict):
            if n not in surface_indexes:
                new_dict.append([name, i])
                if s == n:
                    ns = i
                i += 1
        self.surface_list = new_dict

        self.combo_surface_name.SetItems([n[0] for n in self.surface_list])

        if self.surface_list:
            self.combo_surface_name.SetSelection(ns)

    def OnCloseProject(self):
        self.CloseProject()

    def CloseProject(self):
        n = self.combo_surface_name.GetCount()
        for i in range(n-1, -1, -1):
            self.combo_surface_name.Delete(i)
        self.surface_list = []

    def ChangeSurfaceName(self, index, name):
        self.surface_list[index][0] = name
        self.combo_surface_name.SetString(index, name)

    def InsertNewSurface(self, surface):
        index = surface.index
        name = surface.name
        colour = [int(value*255) for value in surface.colour]
        i = 0
        try:
            i = self.surface_list.index([name, index])
            overwrite = True
        except ValueError:
            overwrite = False

        if overwrite:
            self.surface_list[i] = [name, index]
        else:
            self.surface_list.append([name, index])
            i = len(self.surface_list) - 1

        self.combo_surface_name.SetItems([n[0] for n in self.surface_list])
        self.combo_surface_name.SetSelection(i)
        # transparency = 100*surface.transparency
        # print("Button color: ", colour)
        self.button_colour.SetColour(colour)
        # self.slider_transparency.SetValue(int(transparency))
        #  Publisher.sendMessage('Update surface data', (index))

    def OnComboName(self, evt):
        surface_name = evt.GetString()
        surface_index = evt.GetSelection()
        Publisher.sendMessage('Change surface selected',
                              surface_index=self.surface_list[surface_index][1])

    def OnSelectColour(self, evt):
        colour = [value/255.0 for value in evt.GetValue()]
        Publisher.sendMessage('Set surface colour',
                              surface_index=self.combo_surface_name.GetSelection(),
                              colour=colour)


class NavigationTab(wx.Panel):
    def __init__(self, parent, navigation):
        wx.Panel.__init__(self, parent)

        self.session = ses.Session()
        self.navigation = navigation
        self.sleep_nav = self.navigation.sleep_nav
        self.sleep_coord = const.SLEEP_COORDINATES

        self.LoadConfig()

        text_note = wx.StaticText(
            self, -1, _("Note: Using too low sleep times can result in Invesalius crashing!"))
        # Change sleep pause between navigation loops
        nav_sleep = wx.StaticText(self, -1, _("Navigation Sleep (s):"))
        spin_nav_sleep = wx.SpinCtrlDouble(
            self, -1, "", size=wx.Size(50, 23), inc=0.01)
        spin_nav_sleep.Enable(1)
        spin_nav_sleep.SetRange(0.01, 10.0)
        spin_nav_sleep.SetValue(self.sleep_nav)
        spin_nav_sleep.Bind(wx.EVT_TEXT, partial(
            self.OnSelectNavSleep, ctrl=spin_nav_sleep))
        spin_nav_sleep.Bind(wx.EVT_SPINCTRL, partial(
            self.OnSelectNavSleep, ctrl=spin_nav_sleep))

        # Change sleep pause between coordinate update
        coord_sleep = wx.StaticText(self, -1, _("Coordinate Sleep (s):"))
        spin_coord_sleep = wx.SpinCtrlDouble(
            self, -1, "", size=wx.Size(50, 23), inc=0.01)
        spin_coord_sleep.Enable(1)
        spin_coord_sleep.SetRange(0.01, 10.0)
        spin_coord_sleep.SetValue(self.sleep_coord)
        spin_coord_sleep.Bind(wx.EVT_TEXT, partial(
            self.OnSelectCoordSleep, ctrl=spin_coord_sleep))
        spin_coord_sleep.Bind(wx.EVT_SPINCTRL, partial(
            self.OnSelectCoordSleep, ctrl=spin_coord_sleep))

        line_nav_sleep = wx.BoxSizer(wx.HORIZONTAL)
        line_nav_sleep.AddMany([
            (nav_sleep, 1, wx.EXPAND | wx.GROW | wx.TOP | wx.RIGHT | wx.LEFT, 5),
            (spin_nav_sleep, 0, wx.ALL | wx.EXPAND | wx.GROW, 5)
        ])

        line_coord_sleep = wx.BoxSizer(wx.HORIZONTAL)
        line_coord_sleep.AddMany([
            (coord_sleep, 1, wx.EXPAND | wx.GROW | wx.TOP | wx.RIGHT | wx.LEFT, 5),
            (spin_coord_sleep, 0, wx.ALL | wx.EXPAND | wx.GROW, 5)
        ])

        # Add line sizers into main sizer
        conf_sizer = wx.StaticBoxSizer(
            wx.VERTICAL, self, _("Sleep time configuration"))
        conf_sizer.AddMany([
            (text_note, 0, wx.ALL, 10),
            (line_nav_sleep, 0, wx.GROW | wx.EXPAND |
             wx.LEFT | wx.RIGHT | wx.TOP, 5),
            (line_coord_sleep, 0, wx.GROW | wx.EXPAND |
             wx.LEFT | wx.RIGHT | wx.TOP, 5)
        ])

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.Add(conf_sizer, 0, wx.ALL | wx.EXPAND, 10)
        self.SetSizerAndFit(main_sizer)
        self.Layout()

    def OnSelectNavSleep(self, evt, ctrl):
        self.sleep_nav = ctrl.GetValue()
        self.navigation.UpdateNavSleep(self.sleep_nav)

        self.session.SetConfig('sleep_nav', self.sleep_nav)

    def OnSelectCoordSleep(self, evt, ctrl):
        self.sleep_coord = ctrl.GetValue()
        Publisher.sendMessage('Update coord sleep', data=self.sleep_coord)

        self.session.SetConfig('sleep_coord', self.sleep_nav)

    def LoadConfig(self):
        sleep_nav = self.session.GetConfig('sleep_nav')
        sleep_coord = self.session.GetConfig('sleep_coord')

        if sleep_nav is not None:
            self.sleep_nav = sleep_nav

        if sleep_coord is not None:
            self.sleep_coord = sleep_coord


class ObjectTab(wx.Panel):
    def __init__(self, parent, navigation, tracker, pedal_connector, neuronavigation_api):
        wx.Panel.__init__(self, parent)

        self.session = ses.Session()

        self.coil_list = const.COIL

        self.tracker = tracker
        self.pedal_connector = pedal_connector
        self.neuronavigation_api = neuronavigation_api
        self.navigation = navigation
        self.obj_fiducials = None
        self.obj_orients = None
        self.obj_ref_mode = None
        self.coil_path = None
        self.__bind_events()
        self.timestamp = const.TIMESTAMP
        self.state = self.LoadConfig()

        # Button for creating new stimulator
        tooltip = _("Create new stimulator")
        btn_new = wx.Button(self, -1, _("New"), size=wx.Size(65, 23))
        btn_new.SetToolTip(tooltip)
        btn_new.Enable(1)
        btn_new.Bind(wx.EVT_BUTTON, self.OnCreateNewCoil)
        self.btn_new = btn_new

        # Button for loading stimulator config file
        tooltip = _("Load stimulator configuration file")
        btn_load = wx.Button(self, -1, _("Load"), size=wx.Size(65, 23))
        btn_load.SetToolTip(tooltip)
        btn_load.Enable(1)
        btn_load.Bind(wx.EVT_BUTTON, self.OnLoadCoil)
        self.btn_load = btn_load

        # Save button for saving stimulator config file
        tooltip = _(u"Save stimulator configuration file")
        btn_save = wx.Button(self, -1, _(u"Save"), size=wx.Size(65, 23))
        btn_save.SetToolTip(tooltip)
        btn_save.Enable(1)
        btn_save.Bind(wx.EVT_BUTTON, self.OnSaveCoil)
        self.btn_save = btn_save

        if self.state:
            config_txt = wx.StaticText(
                self, -1, os.path.basename(self.coil_path))
        else:
            config_txt = wx.StaticText(self, -1, "None")

        self.config_txt = config_txt
        lbl = wx.StaticText(self, -1, _("Current Configuration:"))
        lbl.SetFont(wx.Font(9, wx.DEFAULT, wx.NORMAL, wx.BOLD))
        lbl_new = wx.StaticText(
            self, -1, _("Create a new stimulator registration: "))
        lbl_load = wx.StaticText(
            self, -1, _("Load a stimulator registration: "))
        lbl_save = wx.StaticText(
            self, -1, _("Save current stimulator registration: "))

        load_sizer = wx.StaticBoxSizer(
            wx.VERTICAL, self, _("Stimulator registration"))
        inner_load_sizer = wx.FlexGridSizer(2, 4, 5)
        inner_load_sizer.AddMany([
            (lbl, 1, wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5),
            (config_txt, 1, wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5),
            (lbl_new, 1, wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5),
            (btn_new, 1, wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5),
            (lbl_load, 1, wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5),
            (btn_load, 1, wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5),
            (lbl_save, 1, wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5),
            (btn_save, 1, wx.EXPAND | wx.ALIGN_CENTER_VERTICAL, 5),
        ])
        load_sizer.Add(inner_load_sizer, 0, wx.ALL | wx.EXPAND, 10)
        # Change angles threshold
        text_angles = wx.StaticText(self, -1, _("Angle threshold (degrees):"))
        spin_size_angles = wx.SpinCtrlDouble(
            self, -1, "", size=wx.Size(50, 23))
        spin_size_angles.SetRange(0.1, 99)
        spin_size_angles.SetValue(self.angle_threshold)
        spin_size_angles.Bind(wx.EVT_TEXT, partial(
            self.OnSelectAngleThreshold, ctrl=spin_size_angles))
        spin_size_angles.Bind(wx.EVT_SPINCTRL, partial(
            self.OnSelectAngleThreshold, ctrl=spin_size_angles))

        # Change dist threshold
        text_dist = wx.StaticText(self, -1, _("Distance threshold (mm):"))
        spin_size_dist = wx.SpinCtrlDouble(self, -1, "", size=wx.Size(50, 23))
        spin_size_dist.SetRange(0.1, 99)
        spin_size_dist.SetValue(self.distance_threshold)
        spin_size_dist.Bind(wx.EVT_TEXT, partial(
            self.OnSelectDistanceThreshold, ctrl=spin_size_dist))
        spin_size_dist.Bind(wx.EVT_SPINCTRL, partial(
            self.OnSelectDistanceThreshold, ctrl=spin_size_dist))

        # Change timestamp interval
        text_timestamp = wx.StaticText(self, -1, _("Timestamp interval (s):"))
        spin_timestamp_dist = wx.SpinCtrlDouble(
            self, -1, "", size=wx.Size(50, 23), inc=0.1)
        spin_timestamp_dist.SetRange(0.5, 60.0)
        spin_timestamp_dist.SetValue(self.timestamp)
        spin_timestamp_dist.Bind(wx.EVT_TEXT, partial(
            self.OnSelectTimestamp, ctrl=spin_timestamp_dist))
        spin_timestamp_dist.Bind(wx.EVT_SPINCTRL, partial(
            self.OnSelectTimestamp, ctrl=spin_timestamp_dist))
        self.spin_timestamp_dist = spin_timestamp_dist

        # Create a horizontal sizer to threshold configs
        line_angle_threshold = wx.BoxSizer(wx.HORIZONTAL)
        line_angle_threshold.AddMany([
            (text_angles, 1, wx.EXPAND | wx.GROW | wx.TOP | wx.RIGHT | wx.LEFT, 5),
            (spin_size_angles, 0, wx.ALL | wx.EXPAND | wx.GROW, 5)
        ])

        line_dist_threshold = wx.BoxSizer(wx.HORIZONTAL)
        line_dist_threshold.AddMany([
            (text_dist, 1, wx.EXPAND | wx.GROW | wx.TOP | wx.RIGHT | wx.LEFT, 5),
            (spin_size_dist, 0, wx.ALL | wx.EXPAND | wx.GROW, 5)
        ])

        line_timestamp = wx.BoxSizer(wx.HORIZONTAL)
        line_timestamp.AddMany([
            (text_timestamp, 1, wx.EXPAND | wx.GROW |
             wx.TOP | wx.RIGHT | wx.LEFT, 5),
            (spin_timestamp_dist, 0, wx.ALL | wx.EXPAND | wx.GROW, 5)
        ])

        # Add line sizers into main sizer
        conf_sizer = wx.StaticBoxSizer(
            wx.VERTICAL, self, _("Stimulator configuration"))
        conf_sizer.AddMany([
            (line_angle_threshold, 0, wx.GROW |
             wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 20),
            (line_dist_threshold, 0, wx.GROW |
             wx.EXPAND | wx.LEFT | wx.RIGHT | wx.TOP, 20),
            (line_timestamp, 0, wx.GROW | wx.EXPAND |
             wx.LEFT | wx.RIGHT | wx.TOP, 20)
        ])

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.AddMany([
            (load_sizer, 0, wx.ALL | wx.EXPAND, 10),
            (conf_sizer, 0, wx.ALL | wx.EXPAND, 10)
        ])
        self.SetSizerAndFit(main_sizer)
        self.Layout()

    def __bind_events(self):
        Publisher.subscribe(self.OnObjectUpdate, 'Update object registration')

    def LoadConfig(self):
        self.angle_threshold = self.session.GetConfig(
            'angle_threshold') or const.DEFAULT_ANGLE_THRESHOLD
        self.distance_threshold = self.session.GetConfig(
            'distance_threshold') or const.DEFAULT_DISTANCE_THRESHOLD

        state = self.session.GetConfig('navigation')

        if state is not None:
            object_fiducials = np.array(state['object_fiducials'])
            object_orientations = np.array(state['object_orientations'])
            object_reference_mode = state['object_reference_mode']
            object_name = state['object_name'].encode(const.FS_ENCODE)

            self.obj_fiducials, self.obj_orients, self.obj_ref_mode, self.coil_path = object_fiducials, object_orientations, object_reference_mode, object_name

    def OnCreateNewCoil(self, event=None):
        if self.tracker.IsTrackerInitialized():
            dialog = dlg.ObjectCalibrationDialog(
                self.tracker, self.pedal_connector, self.neuronavigation_api)
            try:
                if dialog.ShowModal() == wx.ID_OK:
                    obj_fiducials, obj_orients, obj_ref_mode, coil_path, polydata = dialog.GetValue()

                    self.neuronavigation_api.update_coil_mesh(polydata)

                    if np.isfinite(obj_fiducials).all() and np.isfinite(obj_orients).all():
                        Publisher.sendMessage('Update object registration',
                                              data=(obj_fiducials, obj_orients, obj_ref_mode, coil_path))
                        Publisher.sendMessage('Update status text in GUI',
                                              label=_("Ready"))
                        Publisher.sendMessage(
                            'Configure coil',
                            coil_path=coil_path,
                            polydata=polydata,
                        )

                        # Automatically enable and check 'Track object' checkbox and uncheck 'Disable Volume Camera' checkbox.
                        Publisher.sendMessage(
                            'Enable track object button', enabled=True)
                        Publisher.sendMessage(
                            'Press track object button', pressed=True)

                        Publisher.sendMessage(
                            'Press target mode button', pressed=False)

            except wx._core.PyAssertionError:  # TODO FIX: win64
                pass
            dialog.Destroy()
        else:
            dlg.ShowNavigationTrackerWarning(0, 'choose')

    def OnLoadCoil(self, event=None):
        filename = dlg.ShowLoadSaveDialog(message=_(u"Load object registration"),
                                          wildcard=_("Registration files (*.obr)|*.obr"))
        # data_dir = os.environ.get('OneDrive') + r'\data\dti_navigation\baran\anat_reg_improve_20200609'
        # coil_path = 'magstim_coil_dell_laptop.obr'
        # filename = os.path.join(data_dir, coil_path)

        try:
            if filename:
                with open(filename, 'r') as text_file:
                    data = [s.split('\t') for s in text_file.readlines()]

                registration_coordinates = np.array(
                    data[1:]).astype(np.float32)
                obj_fiducials = registration_coordinates[:, :3]
                obj_orients = registration_coordinates[:, 3:]

                coil_path = data[0][1].encode(const.FS_ENCODE)
                obj_ref_mode = int(data[0][-1])

                if not os.path.exists(coil_path):
                    coil_path = os.path.join(
                        inv_paths.OBJ_DIR, "magstim_fig8_coil.stl")

                polydata = vtk_utils.CreateObjectPolyData(coil_path)
                if polydata:
                    self.neuronavigation_api.update_coil_mesh(polydata)
                else:
                    coil_path = os.path.join(
                        inv_paths.OBJ_DIR, "magstim_fig8_coil.stl")

                Publisher.sendMessage('Update object registration',
                                      data=(obj_fiducials, obj_orients, obj_ref_mode, coil_path))
                Publisher.sendMessage('Update status text in GUI',
                                      label=_("Object file successfully loaded"))
                Publisher.sendMessage(
                    'Configure coil',
                    coil_path=coil_path,
                    polydata=polydata,
                )

                # Automatically enable and check 'Track object' checkbox and uncheck 'Disable Volume Camera' checkbox.
                Publisher.sendMessage(
                    'Enable track object button', enabled=True)
                Publisher.sendMessage(
                    'Press track object button', pressed=True)
                Publisher.sendMessage(
                    'Press target mode button', pressed=False)

                msg = _("Object file successfully loaded")
                wx.MessageBox(msg, _("InVesalius 3"))
        except:
            wx.MessageBox(
                _("Object registration file incompatible."), _("InVesalius 3"))
            Publisher.sendMessage('Update status text in GUI', label="")

    def OnSaveCoil(self, evt):
        obj_fiducials, obj_orients, obj_ref_mode, coil_path = self.navigation.GetObjectRegistration()
        if np.isnan(obj_fiducials).any() or np.isnan(obj_orients).any():
            wx.MessageBox(
                _("Digitize all object fiducials before saving"), _("Save error"))
        else:
            filename = dlg.ShowLoadSaveDialog(message=_(u"Save object registration as..."),
                                              wildcard=_(
                                                  "Registration files (*.obr)|*.obr"),
                                              style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
                                              default_filename="object_registration.obr", save_ext="obr")
            if filename:
                hdr = 'Object' + "\t" + \
                    utils.decode(coil_path, const.FS_ENCODE) + "\t" + \
                    'Reference' + "\t" + str('%d' % obj_ref_mode)
                data = np.hstack([obj_fiducials, obj_orients])
                np.savetxt(filename, data, fmt='%.4f',
                           delimiter='\t', newline='\n', header=hdr)
                wx.MessageBox(_("Object file successfully saved"), _("Save"))

    def OnSelectAngleThreshold(self, evt, ctrl):
        self.angle_threshold = ctrl.GetValue()
        Publisher.sendMessage('Update angle threshold',
                              angle=self.angle_threshold)

        self.session.SetConfig('angle_threshold', self.angle_threshold)

    def OnSelectDistanceThreshold(self, evt, ctrl):
        self.distance_threshold = ctrl.GetValue()
        Publisher.sendMessage('Update distance threshold',
                              dist_threshold=self.distance_threshold)

        self.session.SetConfig('distance_threshold', self.distance_threshold)

    def OnSelectTimestamp(self, evt, ctrl):
        self.timestamp = ctrl.GetValue()

    def OnObjectUpdate(self, data=None):
        self.config_txt.SetLabel(os.path.basename(data[-1]))


class TrackerTab(wx.Panel):
    def __init__(self, parent, tracker, robot):
        wx.Panel.__init__(self, parent)

        self.__bind_events()

        self.tracker = tracker
        self.robot = robot
        self.robot_ip = None
        self.matrix_tracker_to_robot = None
        self.state = self.LoadConfig()

        # ComboBox for spatial tracker device selection
        tracker_options = [_("Select")] + self.tracker.get_trackers()
        select_tracker_elem = wx.ComboBox(self, -1, "", size=(145, -1),
                                          choices=tracker_options, style=wx.CB_DROPDOWN | wx.CB_READONLY)
        tooltip = _("Choose the tracking device")
        select_tracker_elem.SetToolTip(tooltip)
        select_tracker_elem.SetSelection(self.tracker.tracker_id)
        select_tracker_elem.Bind(wx.EVT_COMBOBOX, partial(
            self.OnChooseTracker, ctrl=select_tracker_elem))
        self.select_tracker_elem = select_tracker_elem

        select_tracker_label = wx.StaticText(
            self, -1, _('Choose the tracking device: '))

        # ComboBox for tracker reference mode
        tooltip = _("Choose the navigation reference mode")
        choice_ref = wx.ComboBox(self, -1, "", size=(145, -1),
                                 choices=const.REF_MODE, style=wx.CB_DROPDOWN | wx.CB_READONLY)
        choice_ref.SetSelection(const.DEFAULT_REF_MODE)
        choice_ref.SetToolTip(tooltip)
        choice_ref.Bind(wx.EVT_COMBOBOX, partial(
            self.OnChooseReferenceMode, ctrl=select_tracker_elem))
        self.choice_ref = choice_ref

        choice_ref_label = wx.StaticText(
            self, -1, _('Choose the navigation reference mode: '))

        ref_sizer = wx.FlexGridSizer(rows=2, cols=2, hgap=5, vgap=5)
        ref_sizer.AddMany([
            (select_tracker_label, wx.LEFT),
            (select_tracker_elem, wx.RIGHT),
            (choice_ref_label, wx.LEFT),
            (choice_ref, wx.RIGHT)
        ])
        ref_sizer.Layout()

        sizer = wx.StaticBoxSizer(wx.VERTICAL, self, _("Setup tracker"))
        sizer.Add(ref_sizer, 1, wx.ALL | wx.FIXED_MINSIZE, 20)

        lbl_rob = wx.StaticText(self, -1, _("Select IP for robot device: "))

        # ComboBox for spatial tracker device selection
        tooltip = _("Choose or type the robot IP")
        robot_ip_options = [_("Select robot IP:")] + \
            const.ROBOT_ElFIN_IP + const.ROBOT_DOBOT_IP
        choice_IP = wx.ComboBox(self, -1, "",
                                choices=robot_ip_options, style=wx.CB_DROPDOWN | wx.TE_PROCESS_ENTER)
        choice_IP.SetToolTip(tooltip)
        if self.robot.robot_ip is not None:
            choice_IP.SetSelection(robot_ip_options.index(self.robot.robot_ip))
        else:
            choice_IP.SetSelection(0)
        choice_IP.Bind(wx.EVT_COMBOBOX, partial(
            self.OnChoiceIP, ctrl=choice_IP))
        choice_IP.Bind(wx.EVT_TEXT, partial(self.OnTxt_Ent, ctrl=choice_IP))
        self.choice_IP = choice_IP

        btn_rob = wx.Button(self, -1, _("Connect"))
        btn_rob.SetToolTip("Connect to IP")
        btn_rob.Enable(1)
        btn_rob.Bind(wx.EVT_BUTTON, self.OnRobotConnect)
        self.btn_rob = btn_rob

        status_text = wx.StaticText(self, -1, "Status")
        if self.robot.IsConnected():
            status_text.SetLabelText("Robot is connected!")
            if self.robot.matrix_tracker_to_robot is not None:
                status_text.SetLabelText("Robot is fully setup!")
        else:
            status_text.SetLabelText("Robot is not connected!")
        self.status_text = status_text

        btn_rob_con = wx.Button(self, -1, _("Register"))
        btn_rob_con.SetToolTip("Register robot tracking")
        btn_rob_con.Enable(1)
        btn_rob_con.Bind(wx.EVT_BUTTON, self.OnRobotRegister)
        if self.robot.IsConnected():
            if self.matrix_tracker_to_robot is None:
                btn_rob_con.Show()
            else:
                btn_rob_con.SetLabel("Register Again")
                btn_rob_con.Show()
        else:
            btn_rob_con.Hide()
        self.btn_rob_con = btn_rob_con

        rob_sizer = wx.FlexGridSizer(rows=2, cols=3, hgap=5, vgap=5)
        rob_sizer.AddMany([
            (lbl_rob, 0, wx.LEFT),
            (choice_IP, 1, wx.EXPAND),
            (btn_rob, 0, wx.LEFT | wx.ALIGN_CENTER_HORIZONTAL, 15),
            (status_text, wx.LEFT | wx.ALIGN_CENTER_HORIZONTAL, 15),
            (0, 0),
            (btn_rob_con, 0, wx.LEFT | wx.ALIGN_CENTER_HORIZONTAL, 15)
        ])

        rob_static_sizer = wx.StaticBoxSizer(
            wx.VERTICAL, self, _("Setup robot"))
        rob_static_sizer.Add(rob_sizer, 1, wx.ALL | wx.FIXED_MINSIZE, 20)

        main_sizer = wx.BoxSizer(wx.VERTICAL)
        main_sizer.AddMany([
            (sizer, 0, wx.ALL | wx.EXPAND, 10),
            (rob_static_sizer, 0, wx.ALL | wx.EXPAND, 10)
        ])
        self.SetSizerAndFit(main_sizer)
        self.Layout()

    def __bind_events(self):
        Publisher.subscribe(self.ShowParent, "Show preferences dialog")
        Publisher.subscribe(
            self.OnRobotStatus, "Robot to Neuronavigation: Robot connection status")
        Publisher.subscribe(self.OnSetRobotTransformationMatrix,
                            "Neuronavigation to Robot: Set robot transformation matrix")

    def LoadConfig(self):
        session = ses.Session()
        state = session.GetConfig('robot')

        if state is None:
            return False

        self.robot_ip = state['robot_ip']
        self.matrix_tracker_to_robot = np.array(state['tracker_to_robot'])

        return True

    def OnChooseTracker(self, evt, ctrl):
        if sys.platform == 'darwin':
            wx.CallAfter(self.GetParent().Hide)
        else:
            self.HideParent()
        Publisher.sendMessage('Begin busy cursor')
        Publisher.sendMessage('Update status text in GUI',
                              label=_("Configuring tracker ..."))
        if hasattr(evt, 'GetSelection'):
            choice = evt.GetSelection()
        else:
            choice = None

        self.tracker.DisconnectTracker()
        self.tracker.ResetTrackerFiducials()
        self.tracker.SetTracker(choice)
        Publisher.sendMessage('Update status text in GUI', label=_("Ready"))
        Publisher.sendMessage("Tracker changed")
        ctrl.SetSelection(self.tracker.tracker_id)
        Publisher.sendMessage('End busy cursor')
        if sys.platform == 'darwin':
            wx.CallAfter(self.GetParent().Show)
        else:
            self.ShowParent()

    def OnChooseReferenceMode(self, evt, ctrl):
        # Probably need to refactor object registration as a whole to use the
        # OnChooseReferenceMode function which was used earlier. It can be found in
        # the deprecated code in ObjectRegistrationPanel in task_navigator.py.
        pass

    def HideParent(self):  # hide preferences dialog box
        self.GetGrandParent().Hide()

    def ShowParent(self):  # show preferences dialog box
        self.GetGrandParent().Show()

    def OnTxt_Ent(self, evt, ctrl):
        self.robot_ip = str(ctrl.GetValue())

    def OnChoiceIP(self, evt, ctrl):
        self.robot_ip = ctrl.GetStringSelection()

    def OnRobotConnect(self, evt):
        if self.robot_ip is not None:
            self.status_text.SetLabelText("Trying to connect to robot...")
            self.btn_rob_con.Hide()
            self.robot.SetRobotIP(self.robot_ip)
            Publisher.sendMessage(
                'Neuronavigation to Robot: Connect to robot', robot_IP=self.robot_ip)

    def OnRobotRegister(self, evt):
        if sys.platform == 'darwin':
            wx.CallAfter(self.GetParent().Hide)
        else:
            self.HideParent()
        self.robot.RegisterRobot()
        if sys.platform == 'darwin':
            wx.CallAfter(self.GetParent().Show)
        else:
            self.ShowParent()

    def OnRobotStatus(self, data):
        if data:
            self.status_text.SetLabelText("Setup robot transformation matrix:")
            self.btn_rob_con.Show()

    def OnSetRobotTransformationMatrix(self, data):
        if self.robot.matrix_tracker_to_robot is not None:
            self.status_text.SetLabelText("Robot is fully setup!")
            self.btn_rob_con.SetLabel("Register Again")
            self.btn_rob_con.Show()
            self.btn_rob_con.Layout()
            self.Parent.Update()


class LanguageTab(wx.Panel):
    def __init__(self, parent):
        wx.Panel.__init__(self, parent)

        bsizer = wx.StaticBoxSizer(wx.VERTICAL, self, _("Language"))
        self.lg = lg = ComboBoxLanguage(bsizer.GetStaticBox())
        self.cmb_lang = cmb_lang = lg.GetComboBox()
        text = wx.StaticText(
            bsizer.GetStaticBox(),
            -1,
            _("Language settings will be applied \n the next time InVesalius starts."),
        )
        bsizer.Add(cmb_lang, 0, wx.EXPAND | wx.ALL, 10)
        bsizer.AddSpacer(5)
        bsizer.Add(text, 0, wx.EXPAND | wx.ALL, 10)

        border = wx.BoxSizer()
        border.Add(bsizer, 1, wx.EXPAND | wx.ALL, 10)
        self.SetSizerAndFit(border)
        self.Layout()

    def GetSelection(self):
        selection = self.cmb_lang.GetSelection()
        locales = self.lg.GetLocalesKey()
        options = {const.LANGUAGE: locales[selection]}
        return options

    def LoadSelection(self, values):
        language = values[const.LANGUAGE]
        locales = self.lg.GetLocalesKey()
        selection = locales.index(language)
        self.cmb_lang.SetSelection(int(selection))
