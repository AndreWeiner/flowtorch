"""Module with classes and functions to explore and analyse iPSP data.
"""

# standard library packages
from typing import List
# third party packages
import torch as pt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
# flowtorch packages
from flowtorch.data import PSPDataloader


class PSPExplorer(object):
    """Explore iPSP data interactively.
    """

    def __init__(self, file_path: str):
        """Create an instance from the path to an iPSP file.

        :param file_path: path to iPSP file
        :type file_path: str
        """
        self._file_path = file_path
        self._loader = PSPDataloader(file_path)

    def _aspect_ratio(self, vertices: pt.Tensor) -> dict:
        dx = abs((pt.max(vertices[:, :, 0]) -
                  pt.min(vertices[:, :, 0])).item())
        dy = abs((pt.max(vertices[:, :, 1]) -
                  pt.min(vertices[:, :, 1])).item())
        dz = abs((pt.max(vertices[:, :, 2]) -
                  pt.min(vertices[:, :, 2])).item())
        return {"x": 1.0, "y": dy/dx, "z": dz/dx}

    def _create_time_slider(self, times: List[str]) -> dict:
        """Create a time slider to select snapshots.

        :param times: list of write times
        :type times: List[str]
        :return: slider dictionary
        :rtype: dict
        """
        steps = []
        for i in range(len(times)):
            step = dict(
                method="update",
                args=[{"visible": [False] * len(times)},
                      {"title": "Selected time: {:s}".format(times[i])}],
                label=str(i)
            )
            step["args"][0]["visible"][i] = True
            steps.append(step)
        slider = dict(
            active=0,
            pad={"t": 50},
            steps=steps
        )
        return slider

    def _create_surface_trace(self, field: pt.Tensor, vertices: pt.Tensor,
                              weights: pt.Tensor, every: int, cmin: float, cmax: float):
        return go.Surface(
            x=vertices[::every, ::every, 0],
            y=vertices[::every, ::every, 1],
            z=vertices[::every, ::every, 2],
            surfacecolor=field[::every, ::every] *
            weights[::every, ::every],
            cmin=cmin,
            cmax=cmax
        )

    def _create_surface_layout(self, vertices: pt.Tensor, width: int, times=None) -> go.Layout:
        aspect = self._aspect_ratio(vertices)
        if times is None:
            sliders = None
        else:
            sliders = [self._create_time_slider(times)]
        layout = go.Layout(
            width=width,
            height=int(width*aspect["x"]),
            scene={
                "camera_eye": {"x": 0, "y": -1.0, "z": 0.5},
                "aspectratio": aspect
            },
            sliders=sliders
        )
        return layout

    def interact(self, zone: str, field_name: str, times: list,
                 mask: bool = True, width: int = 1024, every: int = 5,
                 cmin: float = -2, cmax: float = 0.5) -> go.Figure:
        """Show selected snapshots of an iPSP file as interactive surface plot.

        :param zone: name of the zone to display
        :type zone: str
        :param field_name: name of the field to display
        :type field_name: str
        :param times: list of time instances to show
        :type times: list
        :param mask: if True, the data is masked using the mask provided
            in the iPSP dataset, defaults to True
        :type mask: bool, optional
        :param width: width of the plot, defaults to 1024
        :type width: int, optional
        :param every: plot only every nth point, defaults to 5
        :type every: int, optional
        :param cmin: lower bound for color range, defaults to -2
        :type cmin: float, optional
        :param cmax: upper bound for color range, defaults to 0.5
        :type cmax: float, optional
        :return: interactive iPSP surface plot
        :rtype: go.Figure
        """
        self._loader.zone = zone
        vertices = self._loader.vertices
        weights = self._loader.weights
        if not mask:
            weights = 1.0
        fields = self._loader.load_snapshot(field_name, times)
        layout = self._create_surface_layout(vertices, width, times)
        fig = go.Figure(layout=layout)
        for i in range(len(times)):
            field = fields[:, :, i]
            surface = self._create_surface_trace(
                field, vertices, weights, every, cmin, cmax)
            fig.add_trace(surface)
        return fig

    def mean(self, zone: str, field_name: str, times: list,
             mask: bool = True, width: int = 1024, every=5,
             cmin: float = -2, cmax: float = 0.5) -> go.Figure:
        """Show temporal mean of an iPSP file as interactive surface plot.

        :param zone: name of the zone to display
        :type zone: str
        :param field_name: name of the field to display
        :type field_name: str
        :param times: snapshot times over which to compute the mean
        :type times: list
        :param mask: if True, the data is masked using the mask provided
            in the iPSP dataset, defaults to True
        :type mask: bool, optional
        :param width: width of the plot, defaults to 1024
        :type width: int, optional
        :param every: use only every nth point for plotting,
            defaults to 5
        :type every: int, optional
        :param cmin: lower bound for color range, defaults to -2
        :type cmin: float, optional
        :param cmax: upper bound for color range, defaults to 0.5
        :type cmax: float, optional
        :return: interactive iPSP surface plot showing the temporal mean
        :rtype: go.Figure
        """
        self._loader.zone = zone
        vertices = self._loader.vertices
        weights = self._loader.weights
        if not mask:
            weights = 1.0
        fields = self._loader.load_snapshot(field_name, times)
        mean = pt.mean(fields, dim=2)
        layout = self._create_surface_layout(vertices, width)
        fig = go.Figure(layout=layout)
        surface = self._create_surface_trace(
            mean, vertices, weights, every, cmin, cmax)
        fig.add_trace(surface)
        return fig

    def std(self, zone: str, field_name: str, times: list,
            mask: bool = True, width: int = 1024, every=5,
            cmin: float = 0, cmax: float = 0.5) -> go.Figure:
        """Show temporal standard deviation as interactive surface plot.

        :param zone: name of the zone to display
        :type zone: str
        :param field_name: name of the field to display
        :type field_name: str
        :param times: snapshot times over which to compute the standard
            deviation
        :type times: list
        :param mask: if True, the data is masked using the mask provided
            in the iPSP dataset, defaults to True
        :type mask: bool, optional
        :param width: width of the plot, defaults to 1024
        :type width: int, optional
        :param every: use only every nth point for plotting,
            defaults to 5
        :type every: int, optional
        :param cmin: lower bound for color range, defaults to -2
        :type cmin: float, optional
        :param cmax: upper bound for color range, defaults to 0.5
        :type cmax: float, optional
        :return: interactive iPSP surface plot
        :rtype: go.Figure
        """
        self._loader.zone = zone
        vertices = self._loader.vertices
        weights = self._loader.weights
        if not mask:
            weights = 1.0
        fields = self._loader.load_snapshot(field_name, times)
        std = pt.std(fields, dim=2)
        layout = self._create_surface_layout(vertices, width)
        fig = go.Figure(layout=layout)
        surface = self._create_surface_trace(
            std, vertices, weights, every, cmin, cmax)
        fig.add_trace(surface)
        return fig

    @property
    def loader(self) -> PSPDataloader:
        """Get the `PSPdataloader` instance used to access data.

        This property allows accessing the loader's metadata via
        the explorer without having to create a new loader instance.

        :return: PSP dataloader instance
        :rtype: PSPDataloader
        """
        return self._loader
