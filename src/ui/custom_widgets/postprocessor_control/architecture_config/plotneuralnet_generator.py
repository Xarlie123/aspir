"""
PlotNeuralNet-style TikZ code generator for neural network architectures.

Generates self-contained LaTeX/TikZ code for publication-quality diagrams
using the exact same visual style as the PlotNeuralNet library.

Based on: https://github.com/HarisIqbal88/PlotNeuralNet
"""
import os
import subprocess
import tempfile
import logging
import shutil
from typing import List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from .network_visualizer import SemanticBlock, BlockType


@dataclass
class LayoutParams:
    """Parameters controlling diagram layout."""
    base_scale: float = 0.2          # TikZ scale factor
    block_spacing: float = 2.0       # Horizontal spacing between blocks
    encoder_decoder_gap: float = 3.0 # Extra gap before decoder section


# PlotNeuralNet TikZ macros embedded inline (from Box.sty, RightBandedBox.sty, Ball.sty)
TIKZ_MACROS = r'''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Box.sty - Simple 3D box for conv, pool, etc.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\tikzset{Box/.pic={\tikzset{/boxblock/.cd,#1}
        \tikzstyle{box}=[every edge/.append style={pic actions, densely dashed, opacity=.7},fill opacity=\boxopacity, pic actions,fill=\boxfill]

        \pgfmathsetmacro{\y}{\cubey*\boxscale}
        \pgfmathsetmacro{\z}{\cubez*\boxscale}

        %Multiple concatenated boxes
        \foreach[count=\i,%
                 evaluate=\i as \xlabel using {array({\boxlabels},\i-1)},%
                 evaluate=\unscaledx as \k using {\unscaledx*\boxscale+\prev}, remember=\k as \prev (initially 0)]
                 \unscaledx in \cubex
        {
            \pgfmathsetmacro{\x}{\unscaledx*\boxscale}
            \coordinate (a) at (\k-\x , \y/2 , \z/2);
            \coordinate (b) at (\k-\x ,-\y/2 , \z/2);
            \coordinate (c) at (\k    ,-\y/2 , \z/2);
            \coordinate (d) at (\k    , \y/2 , \z/2);
            \coordinate (e) at (\k    , \y/2 ,-\z/2);
            \coordinate (f) at (\k    ,-\y/2 ,-\z/2);
            \coordinate (g) at (\k-\x ,-\y/2 ,-\z/2);
            \coordinate (h) at (\k-\x , \y/2 ,-\z/2);

            \draw [box]
                (d) -- (a) -- (b) -- (c) -- cycle
                (d) -- (a) -- (h) -- (e) -- cycle
                %dotted edges
                (f) edge (g)
                (b) edge (g)
                (h) edge (g)
            ;
            \path (b) edge ["\xlabel"',midway] (c);

            \xdef\LastEastx{\k} %\k persists as \LastEastx after loop
        }%Loop ends
        \draw [box] (d) -- (e) -- (f) -- (c) -- cycle; %East face of last box

        \coordinate (a1) at (0 , \y/2 , \z/2);
        \coordinate (b1) at (0 ,-\y/2 , \z/2);
        \tikzstyle{depthlabel}=[pos=0,text width=14*\z,text centered,sloped]

        \path (c) edge ["\small\zlabel"',depthlabel](f); %depth label
        \path (b1) edge ["\ylabel",midway] (a1);  %height label


        \tikzstyle{captionlabel}=[text width=15*\LastEastx/\boxscale,text centered]
        \path (\LastEastx/2,-\y/2,+\z/2) + (0,-25pt) coordinate (cap)
        edge ["\textcolor{black}{ \bf \boxcaption}"',captionlabel](cap) ; %Block caption/pic object label

        %Define nodes to be used outside on the pic object
        \coordinate (\boxname-west)   at (0,0,0) ;
        \coordinate (\boxname-east)   at (\LastEastx, 0,0) ;
        \coordinate (\boxname-north)  at (\LastEastx/2,\y/2,0);
        \coordinate (\boxname-south)  at (\LastEastx/2,-\y/2,0);
        \coordinate (\boxname-anchor) at (\LastEastx/2, 0,0) ;

        \coordinate (\boxname-near) at (\LastEastx/2,0,\z/2);
        \coordinate (\boxname-far)  at (\LastEastx/2,0,-\z/2);

        \coordinate (\boxname-nearwest) at (0,0,\z/2);
        \coordinate (\boxname-neareast) at (\LastEastx,0,\z/2);
        \coordinate (\boxname-farwest)  at (0,0,-\z/2);
        \coordinate (\boxname-fareast)  at (\LastEastx,0,-\z/2);

        \coordinate (\boxname-northeast) at (\boxname-north-|\boxname-east);
        \coordinate (\boxname-northwest) at (\boxname-north-|\boxname-west);
        \coordinate (\boxname-southeast) at (\boxname-south-|\boxname-east);
        \coordinate (\boxname-southwest) at (\boxname-south-|\boxname-west);

        \coordinate (\boxname-nearnortheast)  at (\LastEastx, \y/2, \z/2);
        \coordinate (\boxname-farnortheast)   at (\LastEastx, \y/2,-\z/2);
        \coordinate (\boxname-nearsoutheast)  at (\LastEastx,-\y/2, \z/2);
        \coordinate (\boxname-farsoutheast)   at (\LastEastx,-\y/2,-\z/2);

        \coordinate (\boxname-nearnorthwest)  at (0, \y/2, \z/2);
        \coordinate (\boxname-farnorthwest)   at (0, \y/2,-\z/2);
        \coordinate (\boxname-nearsouthwest)  at (0,-\y/2, \z/2);
        \coordinate (\boxname-farsouthwest)   at (0,-\y/2,-\z/2);

    },
    /boxblock/.search also={/tikz},
    /boxblock/.cd,
    width/.store        in=\cubex,
    height/.store       in=\cubey,
    depth/.store        in=\cubez,
    scale/.store        in=\boxscale,
    xlabel/.store       in=\boxlabels,
    ylabel/.store       in=\ylabel,
    zlabel/.store       in=\zlabel,
    caption/.store      in=\boxcaption,
    name/.store         in=\boxname,
    fill/.store         in=\boxfill,
    opacity/.store      in=\boxopacity,
    fill={rgb:red,5;green,5;blue,5;white,15},
    opacity=0.4,
    width=2,
    height=13,
    depth=15,
    scale=.2,
    xlabel={{"","","","","","","","","",""}},
    ylabel=,
    zlabel=,
    caption=,
    name=,
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% RightBandedBox.sty - 3D box with right band (for conv+relu, bottleneck)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\tikzset{RightBandedBox/.pic={\tikzset{/block/.cd,#1}

        \tikzstyle{box}=[every edge/.append style={pic actions, densely dashed, opacity=.7},fill opacity=\rbopacity, pic actions,fill=\rbfill]

        \tikzstyle{band}=[every edge/.append style={pic actions, densely dashed, opacity=.7},fill opacity=\rbbandopacity, pic actions,fill=\rbbandfill,draw=\rbbandfill]

        \pgfmathsetmacro{\y}{\rbcubey*\rbscale}
        \pgfmathsetmacro{\z}{\rbcubez*\rbscale}

        %Multiple concatenated boxes
        \foreach[count=\i,%
                 evaluate=\i as \xlabel using {array({\rbboxlabels},\i-1)},%
                 evaluate=\unscaledx as \k using {\unscaledx*\rbscale+\prev}, remember=\k as \prev (initially 0)]
                 \unscaledx in \rbcubex
        {
            \pgfmathsetmacro{\x}{\unscaledx*\rbscale}
            \coordinate (a)     at (\k-\x   , \y/2 , \z/2);
            \coordinate (art)   at (\k-\x/3 , \y/2 , \z/2); %a_right_third
            \coordinate (b)     at (\k-\x   ,-\y/2 , \z/2);
            \coordinate (brt)   at (\k-\x/3 ,-\y/2 , \z/2); %b_right_third
            \coordinate (c)     at (\k      ,-\y/2 , \z/2);
            \coordinate (d)     at (\k      , \y/2 , \z/2);
            \coordinate (e)     at (\k      , \y/2 ,-\z/2);
            \coordinate (f)     at (\k      ,-\y/2 ,-\z/2);
            \coordinate (g)     at (\k-\x   ,-\y/2 ,-\z/2);
            \coordinate (h)     at (\k-\x   , \y/2 ,-\z/2);
            \coordinate (hrt)   at (\k-\x/3 , \y/2 ,-\z/2); %h_right_third

            %fill box color
            \draw [box]
                (d) -- (a) -- (b) -- (c) -- cycle
                (d) -- (a) -- (h) -- (e) -- cycle;
            %dotted edges
            \draw [box]
                (f) edge (g)
                (b) edge (g)
                (h) edge (g);
            %fill band color
            \draw [band]
                (d) -- (art) -- (brt) -- (c) -- cycle
                (d) -- (art) -- (hrt) -- (e) -- cycle;
            %draw edges again which were covered by band
            \draw [box,fill opacity=0]
                (d) -- (a) -- (b) -- (c) -- cycle
                (d) -- (a) -- (h) -- (e) -- cycle;

            \path (b) edge ["\xlabel"',midway] (c);

            \xdef\LastEastx{\k} %\k persists as \LastEastx after loop
        }%Loop ends
        \draw [box] (d) -- (e) -- (f) -- (c) -- cycle; %East face of last box
        \draw [band] (d) -- (e) -- (f) -- (c) -- cycle; %East face of last box
        \draw [pic actions] (d) -- (e) -- (f) -- (c) -- cycle; %East face edges of last box

        \coordinate (a1) at (0 , \y/2 , \z/2);
        \coordinate (b1) at (0 ,-\y/2 , \z/2);
        \tikzstyle{depthlabel}=[pos=0,text width=14*\z,text centered,sloped]

        \path (c) edge ["\small\rbzlabel"',depthlabel](f); %depth label
        \path (b1) edge ["\rbylabel",midway] (a1);  %height label

        \tikzstyle{captionlabel}=[text width=15*\LastEastx/\rbscale,text centered]
        \path (\LastEastx/2,-\y/2,+\z/2) + (0,-25pt) coordinate (cap)
        edge ["\textcolor{black}{ \bf \rbcaption}"',captionlabel] (cap); %Block caption/pic object label

        %Define nodes to be used outside on the pic object
        \coordinate (\rbname-west)   at (0,0,0) ;
        \coordinate (\rbname-east)   at (\LastEastx, 0,0) ;
        \coordinate (\rbname-north)  at (\LastEastx/2,\y/2,0);
        \coordinate (\rbname-south)  at (\LastEastx/2,-\y/2,0);
        \coordinate (\rbname-anchor) at (\LastEastx/2, 0,0) ;

        \coordinate (\rbname-near) at (\LastEastx/2,0,\z/2);
        \coordinate (\rbname-far)  at (\LastEastx/2,0,-\z/2);

        \coordinate (\rbname-nearwest) at (0,0,\z/2);
        \coordinate (\rbname-neareast) at (\LastEastx,0,\z/2);
        \coordinate (\rbname-farwest)  at (0,0,-\z/2);
        \coordinate (\rbname-fareast)  at (\LastEastx,0,-\z/2);

        \coordinate (\rbname-northeast) at (\rbname-north-|\rbname-east);
        \coordinate (\rbname-northwest) at (\rbname-north-|\rbname-west);
        \coordinate (\rbname-southeast) at (\rbname-south-|\rbname-east);
        \coordinate (\rbname-southwest) at (\rbname-south-|\rbname-west);

        \coordinate (\rbname-nearnortheast)  at (\LastEastx, \y/2, \z/2);
        \coordinate (\rbname-farnortheast)   at (\LastEastx, \y/2,-\z/2);
        \coordinate (\rbname-nearsoutheast)  at (\LastEastx,-\y/2, \z/2);
        \coordinate (\rbname-farsoutheast)   at (\LastEastx,-\y/2,-\z/2);

        \coordinate (\rbname-nearnorthwest)  at (0, \y/2, \z/2);
        \coordinate (\rbname-farnorthwest)   at (0, \y/2,-\z/2);
        \coordinate (\rbname-nearsouthwest)  at (0,-\y/2, \z/2);
        \coordinate (\rbname-farsouthwest)   at (0,-\y/2,-\z/2);
    },
    /block/.search also={/tikz},
    /block/.cd,
    width/.store        in=\rbcubex,
    height/.store       in=\rbcubey,
    depth/.store        in=\rbcubez,
    scale/.store        in=\rbscale,
    xlabel/.store       in=\rbboxlabels,
    ylabel/.store       in=\rbylabel,
    zlabel/.store       in=\rbzlabel,
    caption/.store      in=\rbcaption,
    name/.store         in=\rbname,
    fill/.store         in=\rbfill,
    bandfill/.store     in=\rbbandfill,
    opacity/.store      in=\rbopacity,
    bandopacity/.store  in=\rbbandopacity,
    fill={rgb:red,5;green,5;blue,5;white,15},
    bandfill={rgb:red,5;green,5;blue,5;white,5},
    opacity=0.4,
    bandopacity=0.6,
    width=2,
    height=13,
    depth=15,
    scale=.2,
    xlabel={{"","","","","","","","","",""}},
    ylabel=,
    zlabel=,
    caption=,
    name=,
}

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Ball.sty - Sphere for sum/concat operations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
\tikzset{Ball/.pic={\tikzset{/sphere/.cd,#1}

\pgfmathsetmacro{\r}{\ballradius*\ballscale}

\shade[ball color=\ballfill,opacity=\ballopacity] (0,0,0) circle (\r);
\draw (0,0,0) circle [radius=\r] node[scale=4*\r] {\balllogo};

\coordinate (\ballname-anchor) at ( 0 , 0  , 0) ;
\coordinate (\ballname-east)   at ( \r, 0  , 0) ;
\coordinate (\ballname-west)   at (-\r, 0  , 0) ;
\coordinate (\ballname-north)  at ( 0 , \r , 0) ;
\coordinate (\ballname-south)  at ( 0 , -\r, 0) ;

\path (\ballname-south) + (0,-20pt) coordinate (caption-node)
edge ["\textcolor{black}{\bf \ballcaption}"'] (caption-node); %Ball caption

},
/sphere/.search also={/tikz},
/sphere/.cd,
radius/.store       in=\ballradius,
scale/.store        in=\ballscale,
caption/.store      in=\ballcaption,
name/.store         in=\ballname,
fill/.store         in=\ballfill,
logo/.store         in=\balllogo,
opacity/.store      in=\ballopacity,
logo=$\Sigma$,
fill=green,
opacity=0.10,
scale=0.2,
radius=0.5,
caption=,
name=,
}
'''


class PlotNeuralNetGenerator:
    """
    Generates TikZ/LaTeX code from SemanticBlock architecture.

    Uses the exact visual style of PlotNeuralNet library by embedding
    the TikZ macros inline.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.layout = LayoutParams()
        self._temp_dir: Optional[str] = None

    def is_available(self) -> bool:
        """Check if pdflatex is available."""
        try:
            result = subprocess.run(
                ['pdflatex', '--version'],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    def generate(self,
                 blocks: List[SemanticBlock],
                 skip_connections: List[Tuple[int, int]],
                 model_name: str = "Neural Network",
                 input_image: Optional[np.ndarray] = None,
                 output_image: Optional[np.ndarray] = None,
                 output_path: Optional[str] = None,
                 keep_tex: bool = False,
                 colormap: Optional[str] = None) -> Optional[str]:
        """
        Generate PDF visualization from blocks.

        Args:
            blocks: List of SemanticBlocks from NetworkVisualizer
            skip_connections: List of (encoder_idx, decoder_idx) tuples
            model_name: Title for the diagram
            input_image: Optional sample input image
            output_image: Optional sample output image
            output_path: Path for output PDF (uses temp if None)
            keep_tex: Keep the .tex file alongside PDF
            colormap: Optional colormap name for grayscale images

        Returns:
            Path to generated PDF, or None if failed
        """
        if not self.is_available():
            self.logger.error("pdflatex not found. Install texlive-latex-base.")
            return None

        if not blocks:
            self.logger.error("No blocks to visualize")
            return None

        # Create temp directory for compilation
        self._temp_dir = tempfile.mkdtemp(prefix="plotnn_")

        try:
            # Save images if provided
            input_img_path = None
            output_img_path = None

            if input_image is not None:
                input_img_path = self._save_image(input_image, "input.png", colormap)
            if output_image is not None:
                output_img_path = self._save_image(output_image, "output.png", colormap)

            # Generate TikZ code
            tikz_code = self._generate_tikz(
                blocks, skip_connections, model_name,
                input_img_path, output_img_path
            )

            # Write .tex file
            tex_path = os.path.join(self._temp_dir, "network.tex")
            with open(tex_path, 'w') as f:
                f.write(tikz_code)

            self.logger.debug(f"Generated TikZ code ({len(tikz_code)} chars)")

            # Compile LaTeX
            pdf_path = self._compile_latex(tex_path)

            if pdf_path and output_path:
                # Copy to requested location
                shutil.copy2(pdf_path, output_path)
                if keep_tex:
                    tex_out = output_path.replace('.pdf', '.tex')
                    shutil.copy2(tex_path, tex_out)
                pdf_path = output_path
                # Cleanup temp since we copied to output_path
                self._cleanup_temp()

            return pdf_path

        except Exception as e:
            self.logger.error(f"Generation failed: {e}", exc_info=True)
            self._cleanup_temp()
            return None

    def _cleanup_temp(self):
        """Clean up temporary directory."""
        if self._temp_dir:
            try:
                shutil.rmtree(self._temp_dir)
            except Exception:
                pass
            self._temp_dir = None

    def cleanup(self):
        """Public method to clean up temporary files. Call when done with PDF."""
        self._cleanup_temp()

    def _save_image(self, img_array: np.ndarray, filename: str,
                     colormap: Optional[str] = None) -> str:
        """Save numpy array as image in temp directory.

        Args:
            img_array: Image data as numpy array
            filename: Output filename
            colormap: Optional colormap name (Gray, Viridis, Jet, Hot, Inferno, Plasma)

        Returns:
            Path to saved image file
        """
        path = os.path.join(self._temp_dir, filename)

        # Convert to 2D if needed
        if img_array.ndim == 3:
            if img_array.shape[0] == 1:
                # (1, H, W) -> (H, W)
                img_array = img_array[0]
            elif img_array.shape[2] == 1:
                # (H, W, 1) -> (H, W)
                img_array = img_array[:, :, 0]
            elif img_array.shape[0] in [3, 4]:
                # (C, H, W) -> (H, W, C)
                img_array = np.transpose(img_array, (1, 2, 0))

        # Normalize to [0, 1] range
        img_min = img_array.min()
        img_max = img_array.max()

        if img_max > img_min:
            img_normalized = (img_array - img_min) / (img_max - img_min)
        else:
            # Constant image - use mid-gray
            img_normalized = np.full(img_array.shape, 0.5)

        # Apply colormap if specified and not grayscale
        if colormap and colormap.lower() != 'gray' and img_normalized.ndim == 2:
            # Map colormap name to matplotlib colormap
            colormap_mapping = {
                'viridis': 'viridis',
                'jet': 'jet',
                'hot': 'hot',
                'inferno': 'inferno',
                'plasma': 'plasma',
            }
            cmap_name = colormap_mapping.get(colormap.lower(), 'viridis')
            cmap = plt.get_cmap(cmap_name)

            # Apply colormap (returns RGBA)
            img_colored = cmap(img_normalized)
            # Convert RGBA to RGB and scale to 0-255
            img_uint8 = (img_colored[:, :, :3] * 255).astype(np.uint8)
        else:
            # Grayscale
            img_uint8 = (img_normalized * 255).astype(np.uint8)

        img = Image.fromarray(img_uint8)
        img.save(path)
        return path

    def _generate_tikz(self,
                       blocks: List[SemanticBlock],
                       skip_connections: List[Tuple[int, int]],
                       model_name: str,
                       input_img_path: Optional[str],
                       output_img_path: Optional[str]) -> str:
        """Generate complete TikZ/LaTeX document in PlotNeuralNet style."""
        lines = []

        # Document header
        lines.append(self._to_head())
        lines.append(self._to_colors())
        lines.append(TIKZ_MACROS)
        lines.append(self._to_begin())

        # Calculate input block size to match image size
        input_block_size = self._calculate_input_block_size(blocks)

        # Input image (sized to match input layer)
        if input_img_path:
            img_name = os.path.basename(input_img_path)
            # Convert TikZ units to cm: height * scale = actual size in TikZ units
            # TikZ uses 1cm per unit by default, so the image should match
            img_size_cm = input_block_size * self.layout.base_scale
            lines.append(self._to_input(img_name, "(-3,0,0)",
                                       width=img_size_cm, height=img_size_cm))

        # Generate architecture
        arch_code = self._generate_architecture(blocks, skip_connections)
        lines.extend(arch_code)

        # Output image (sized to match output layer)
        if output_img_path:
            img_name = os.path.basename(output_img_path)
            # Calculate output block size
            output_block_size = self._calculate_output_block_size(blocks)
            output_img_size_cm = output_block_size * self.layout.base_scale
            # Position after last block using path calculation for correct 3D placement
            lines.append(self._to_output_image(img_name, output_img_size_cm))

        lines.append(self._to_end())

        return "\n".join(lines)

    def _to_head(self) -> str:
        """Generate LaTeX document preamble."""
        return r'''\documentclass[border=8pt, multi, tikz]{standalone}
\usepackage{tikz}
\usetikzlibrary{quotes,arrows.meta}
\usetikzlibrary{positioning}
\usetikzlibrary{3d}
\usetikzlibrary{calc}
'''

    def _to_colors(self) -> str:
        """Define PlotNeuralNet color scheme."""
        return r'''
\def\ConvColor{rgb:yellow,5;red,2.5;white,5}
\def\ConvReluColor{rgb:yellow,5;red,5;white,5}
\def\PoolColor{rgb:red,1;black,0.3}
\def\UnpoolColor{rgb:blue,2;green,1;black,0.3}
\def\FcColor{rgb:blue,5;red,2.5;white,5}
\def\FcReluColor{rgb:blue,5;red,5;white,4}
\def\SoftmaxColor{rgb:magenta,5;black,7}
\def\SumColor{rgb:blue,5;green,15}
\def\InputColor{rgb:blue,5;white,5}
\def\OutputColor{rgb:green,5;white,5}
\def\BottleneckColor{rgb:orange,5;red,2;white,3}

\def\edgecolor{rgb:blue,4;red,1;green,4;black,3}
\newcommand{\midarrow}{\tikz \draw[-Stealth,line width=0.8mm,draw=\edgecolor] (-0.3,0) -- ++(0.3,0);}
\newcommand{\copymidarrow}{\tikz \draw[-Stealth,line width=0.8mm,draw={rgb:blue,4;red,1;green,1;black,3}] (-0.3,0) -- ++(0.3,0);}
'''

    def _to_begin(self) -> str:
        """Begin TikZ picture."""
        return r'''
\begin{document}
\begin{tikzpicture}
\tikzstyle{connection}=[ultra thick,every node/.style={sloped,allow upside down},draw=\edgecolor,opacity=0.7]
\tikzstyle{copyconnection}=[ultra thick,every node/.style={sloped,allow upside down},draw={rgb:blue,4;red,1;green,1;black,3},opacity=0.7]
'''

    def _to_end(self) -> str:
        """End TikZ picture."""
        return r'''
\end{tikzpicture}
\end{document}
'''

    def _to_input(self, pathfile: str, to: str, width: float = 4, height: float = 4,
                  name: str = "input_img", offset: str = "(0,0,0)") -> str:
        """Generate input image node."""
        return rf'''
\node[canvas is zy plane at x=0, shift={{{offset}}}] ({name}) at {to} {{\includegraphics[width={width:.2f}cm,height={height:.2f}cm]{{{pathfile}}}}};
'''

    def _to_output_image(self, pathfile: str, size_cm: float, spacing: float = 3.0) -> str:
        """Generate output image node positioned after the last layer.

        Args:
            pathfile: Path to the image file
            size_cm: Size of the image in cm (width and height)
            spacing: Horizontal spacing from the last layer in TikZ units
        """
        # Calculate x position: end-east x + spacing
        # Use TikZ let operation to extract coordinate and position correctly
        return rf'''
% Output image - positioned to the right of the network
\path (end-east);
\pgfgetlastxy{{\XCoord}}{{\YCoord}}
\node[canvas is zy plane at x=0] (output_img) at (\XCoord/1cm + {spacing}, 0, 0) {{\includegraphics[width={size_cm:.2f}cm,height={size_cm:.2f}cm]{{{pathfile}}}}};
'''

    def _calculate_input_block_size(self, blocks: List[SemanticBlock]) -> float:
        """Calculate the size of the input block for image matching.

        Returns the height/depth value used for the input layer in TikZ units.
        """
        from .network_visualizer import BlockType

        # Find input block or first encoder block
        input_block = None
        for block in blocks:
            if block.block_type == BlockType.INPUT:
                input_block = block
                break

        if input_block is None and blocks:
            input_block = blocks[0]

        if input_block is None:
            return 50.0  # Default size

        # Calculate size using same logic as _generate_architecture
        spatial_sizes = [b.spatial_size for b in blocks if b.spatial_size > 0]
        channel_counts = [b.out_channels for b in blocks if b.out_channels > 0]

        spatial_varies = len(set(spatial_sizes)) > 2 if spatial_sizes else False
        max_channels = max(channel_counts) if channel_counts else 64
        min_channels = min(channel_counts) if channel_counts else 64
        max_spatial = max(spatial_sizes) if spatial_sizes else 1

        if spatial_varies:
            # CNN-like: height/depth based on spatial size
            if max_spatial > 1:
                normalized = input_block.spatial_size / max_spatial
                height = 15 + normalized * 35
            else:
                height = 30
        else:
            # Autoencoder/MLP: height/depth based on channel count
            if max_channels > min_channels:
                normalized = (input_block.out_channels - min_channels) / (max_channels - min_channels)
                height = 15 + normalized * 35
            else:
                height = 30

        return height

    def _calculate_output_block_size(self, blocks: List[SemanticBlock]) -> float:
        """Calculate the size of the output block for image matching.

        Returns the height/depth value used for the output layer in TikZ units.
        """
        from .network_visualizer import BlockType

        # Find output block or last decoder block
        output_block = None
        for block in reversed(blocks):
            if block.block_type == BlockType.OUTPUT:
                output_block = block
                break

        if output_block is None and blocks:
            # Use last block if no explicit output block
            output_block = blocks[-1]

        if output_block is None:
            return 50.0  # Default size

        # Calculate size using same logic as _generate_architecture
        spatial_sizes = [b.spatial_size for b in blocks if b.spatial_size > 0]
        channel_counts = [b.out_channels for b in blocks if b.out_channels > 0]

        spatial_varies = len(set(spatial_sizes)) > 2 if spatial_sizes else False
        max_channels = max(channel_counts) if channel_counts else 64
        min_channels = min(channel_counts) if channel_counts else 64
        max_spatial = max(spatial_sizes) if spatial_sizes else 1

        if spatial_varies:
            # CNN-like: height/depth based on spatial size
            if max_spatial > 1:
                normalized = output_block.spatial_size / max_spatial
                height = 15 + normalized * 35
            else:
                height = 30
        else:
            # Autoencoder/MLP: height/depth based on channel count
            if max_channels > min_channels:
                normalized = (output_block.out_channels - min_channels) / (max_channels - min_channels)
                height = 15 + normalized * 35
            else:
                height = 30

        return height

    def _to_conv(self, name: str, s_filer: int = 256, n_filer: int = 64,
                 offset: str = "(0,0,0)", to: str = "(0,0,0)",
                 width: float = 2, height: float = 40, depth: float = 40,
                 caption: str = " ") -> str:
        """Generate Conv layer (single box)."""
        return rf'''
\pic[shift={{{offset}}}] at {to}
    {{Box={{
        name={name},
        caption={caption},
        xlabel={{{{{n_filer}, }}}},
        zlabel={s_filer},
        fill=\ConvColor,
        height={height},
        width={width},
        depth={depth}
        }}
    }};
'''

    def _to_conv_conv_relu(self, name: str, s_filer: int = 256,
                           n_filer: Tuple[int, int] = (64, 64),
                           offset: str = "(0,0,0)", to: str = "(0,0,0)",
                           width: Tuple[float, float] = (2, 2),
                           height: float = 40, depth: float = 40,
                           caption: str = " ", color: str = "\\ConvColor",
                           band_color: str = "\\ConvReluColor") -> str:
        """Generate ConvConvRelu layer (banded box for bottleneck)."""
        return rf'''
\pic[shift={{{offset}}}] at {to}
    {{RightBandedBox={{
        name={name},
        caption={caption},
        xlabel={{{{{n_filer[0]}, {n_filer[1]} }}}},
        zlabel={s_filer},
        fill={color},
        bandfill={band_color},
        height={height},
        width={{ {width[0]} , {width[1]} }},
        depth={depth}
        }}
    }};
'''

    def _to_pool(self, name: str, offset: str = "(0,0,0)", to: str = "(0,0,0)",
                 width: float = 1, height: float = 32, depth: float = 32,
                 opacity: float = 0.5, caption: str = " ") -> str:
        """Generate Pool layer."""
        return rf'''
\pic[shift={{{offset}}}] at {to}
    {{Box={{
        name={name},
        caption={caption},
        fill=\PoolColor,
        opacity={opacity},
        height={height},
        width={width},
        depth={depth}
        }}
    }};
'''

    def _to_unpool(self, name: str, offset: str = "(0,0,0)", to: str = "(0,0,0)",
                   width: float = 1, height: float = 32, depth: float = 32,
                   opacity: float = 0.5, caption: str = " ") -> str:
        """Generate UnPool/Upsample layer."""
        return rf'''
\pic[shift={{{offset}}}] at {to}
    {{Box={{
        name={name},
        caption={caption},
        fill=\UnpoolColor,
        opacity={opacity},
        height={height},
        width={width},
        depth={depth}
        }}
    }};
'''

    def _to_connection(self, of: str, to: str) -> str:
        """Generate connection arrow between layers."""
        return rf'''
\draw [connection]  ({of}-east) -- node {{\midarrow}} ({to}-west);
'''

    def _to_skip(self, of: str, to: str, pos: float = 1.25) -> str:
        """Generate skip connection (curved path over the top)."""
        return rf'''
\path ({of}-southeast) -- ({of}-northeast) coordinate[pos={pos}] ({of}-top) ;
\path ({to}-south) -- ({to}-north) coordinate[pos={pos}] ({to}-top) ;
\draw [copyconnection]  ({of}-northeast)
-- node {{\copymidarrow}}({of}-top)
-- node {{\copymidarrow}}({to}-top)
-- node {{\copymidarrow}} ({to}-north);
'''

    def _generate_architecture(self, blocks: List[SemanticBlock],
                               skip_connections: List[Tuple[int, int]]) -> List[str]:
        """Generate TikZ code for entire architecture."""
        import math
        lines = []
        block_names = []

        # Analyze blocks to determine sizing strategy
        spatial_sizes = [b.spatial_size for b in blocks if b.spatial_size > 0]
        channel_counts = [b.out_channels for b in blocks if b.out_channels > 0]

        # Check if spatial sizes vary (CNN-like) or are constant (autoencoder/MLP)
        spatial_varies = len(set(spatial_sizes)) > 2 if spatial_sizes else False
        max_channels = max(channel_counts) if channel_counts else 64
        min_channels = min(channel_counts) if channel_counts else 64
        max_spatial = max(spatial_sizes) if spatial_sizes else 1

        def calc_size(block: SemanticBlock) -> Tuple[float, float]:
            """Calculate height and depth from block properties."""
            if spatial_varies:
                # CNN-like: height/depth based on spatial size
                # Normalize to range [15, 50]
                if max_spatial > 1:
                    normalized = block.spatial_size / max_spatial
                    height = 15 + normalized * 35
                else:
                    height = 30
            else:
                # Autoencoder/MLP: height/depth based on channel count
                # More channels = larger block (shows the compression in middle)
                if max_channels > min_channels:
                    normalized = (block.out_channels - min_channels) / (max_channels - min_channels)
                    height = 15 + normalized * 35
                else:
                    height = 30
            depth = height  # Keep aspect ratio square
            return height, depth

        def calc_width(block: SemanticBlock) -> float:
            """Calculate width (thickness) from channel count."""
            # Normalize width based on channel count relative to max
            if max_channels > 1:
                normalized = math.log2(max(block.out_channels, 1)) / math.log2(max_channels)
                return max(1.5, min(5, 1.5 + normalized * 3.5))
            return 2.5

        def get_caption(block: SemanticBlock) -> str:
            """Get display caption for block type."""
            caption_map = {
                BlockType.INPUT: "Input",
                BlockType.CONV_BLOCK: "Conv",
                BlockType.POOL: "Pool",
                BlockType.UPSAMPLE: "Upsample",
                BlockType.BOTTLENECK: "Bottleneck",
                BlockType.OUTPUT: "Output",
                BlockType.FC: "FC",
                BlockType.RESIDUAL: "Res",
                BlockType.ATTENTION: "Attention",
            }
            return caption_map.get(block.block_type, "")

        # Group blocks into encoder, bottleneck, decoder
        encoder_blocks = []
        decoder_blocks = []
        bottleneck_block = None
        bottleneck_idx = -1

        for i, block in enumerate(blocks):
            if block.block_type == BlockType.BOTTLENECK:
                bottleneck_block = (i, block)
                bottleneck_idx = i
            elif block.is_encoder:
                encoder_blocks.append((i, block))
            else:
                decoder_blocks.append((i, block))

        # Build name mapping for skip connections
        idx_to_name = {}

        # Generate encoder
        prev_name = None
        for i, (idx, block) in enumerate(encoder_blocks):
            height, depth = calc_size(block)
            width = calc_width(block)
            name = f"enc{i}"
            idx_to_name[idx] = name
            block_names.append(name)

            caption = get_caption(block)

            if block.block_type == BlockType.POOL:
                # Pool layer
                if prev_name:
                    offset = "(0,0,0)"
                    to = f"({prev_name}-east)"
                else:
                    offset = "(0,0,0)"
                    to = "(0,0,0)"
                lines.append(self._to_pool(
                    name=name,
                    offset=offset,
                    to=to,
                    height=height * 0.75,
                    depth=depth * 0.75,
                    width=1,
                    caption=caption
                ))
            else:
                # Conv block
                if prev_name:
                    offset = f"({self.layout.block_spacing},0,0)"
                    to = f"({prev_name}-east)"
                else:
                    offset = "(0,0,0)"
                    to = "(0,0,0)"
                lines.append(self._to_conv_conv_relu(
                    name=name,
                    s_filer=block.spatial_size,
                    n_filer=(block.out_channels, block.out_channels),
                    offset=offset,
                    to=to,
                    width=(width, width),
                    height=height,
                    depth=depth,
                    caption=caption
                ))

            # Connection from previous
            if prev_name and block.block_type != BlockType.POOL:
                lines.append(self._to_connection(prev_name, name))

            prev_name = name

        # Generate bottleneck
        if bottleneck_block:
            idx, block = bottleneck_block
            height, depth = calc_size(block)
            width = calc_width(block)
            name = "bottleneck"
            idx_to_name[idx] = name
            block_names.append(name)

            offset = f"({self.layout.encoder_decoder_gap},0,0)"
            to = f"({prev_name}-east)" if prev_name else "(0,0,0)"

            lines.append(self._to_conv_conv_relu(
                name=name,
                s_filer=block.spatial_size,
                n_filer=(block.out_channels, block.out_channels),
                offset=offset,
                to=to,
                width=(width, width),
                height=height,
                depth=depth,
                caption="Bottleneck",
                color="\\BottleneckColor",
                band_color="\\BottleneckColor"
            ))
            if prev_name:
                lines.append(self._to_connection(prev_name, name))
            prev_name = name

        # Generate decoder
        for i, (idx, block) in enumerate(decoder_blocks):
            height, depth = calc_size(block)
            width = calc_width(block)
            name = f"dec{i}"
            idx_to_name[idx] = name
            block_names.append(name)

            caption = get_caption(block)

            if block.block_type == BlockType.UPSAMPLE:
                # Upsample layer - placed close to previous (like Pool in encoder)
                offset = "(0,0,0)"
                to = f"({prev_name}-east)" if prev_name else "(0,0,0)"
                lines.append(self._to_unpool(
                    name=name,
                    offset=offset,
                    to=to,
                    height=height,
                    depth=depth,
                    width=1,
                    caption=caption
                ))
            else:
                # Conv block - spaced from previous (like Conv in encoder)
                offset = f"({self.layout.block_spacing},0,0)"
                to = f"({prev_name}-east)" if prev_name else "(0,0,0)"
                lines.append(self._to_conv_conv_relu(
                    name=name,
                    s_filer=block.spatial_size,
                    n_filer=(block.out_channels, block.out_channels),
                    offset=offset,
                    to=to,
                    width=(width, width),
                    height=height,
                    depth=depth,
                    caption=caption
                ))

            # Connection from previous
            if prev_name and block.block_type != BlockType.UPSAMPLE:
                lines.append(self._to_connection(prev_name, name))

            prev_name = name

        # Mark last block as 'end' for output image positioning
        if prev_name:
            lines.append(rf"\coordinate (end-east) at ({prev_name}-east);")

        # Generate skip connections
        for enc_idx, dec_idx in skip_connections:
            enc_name = idx_to_name.get(enc_idx)
            dec_name = idx_to_name.get(dec_idx)
            if enc_name and dec_name:
                lines.append(self._to_skip(enc_name, dec_name))

        return lines

    def _compile_latex(self, tex_path: str, timeout: int = 60) -> Optional[str]:
        """Compile LaTeX to PDF using pdflatex."""
        work_dir = os.path.dirname(tex_path)
        tex_name = os.path.basename(tex_path)

        try:
            # Run pdflatex (twice for proper references)
            for i in range(2):
                result = subprocess.run(
                    ['pdflatex', '-interaction=nonstopmode', tex_name],
                    cwd=work_dir,
                    capture_output=True,
                    timeout=timeout,
                    text=True
                )

                if result.returncode != 0:
                    self.logger.warning(f"pdflatex pass {i+1} returned {result.returncode}")
                    # Log last part of output for debugging
                    if result.stdout:
                        self.logger.debug(f"pdflatex output:\n{result.stdout[-1500:]}")

            # Check for PDF output
            pdf_path = tex_path.replace('.tex', '.pdf')
            if os.path.exists(pdf_path):
                self.logger.info(f"Generated PDF: {pdf_path}")
                return pdf_path
            else:
                self.logger.error("PDF not generated")
                return None

        except subprocess.TimeoutExpired:
            self.logger.error(f"pdflatex timed out after {timeout}s")
            return None
        except Exception as e:
            self.logger.error(f"Compilation error: {e}")
            return None

    def get_tex_source(self,
                       blocks: List[SemanticBlock],
                       skip_connections: List[Tuple[int, int]],
                       model_name: str = "Neural Network") -> str:
        """Get the TikZ/LaTeX source code without compiling."""
        return self._generate_tikz(blocks, skip_connections, model_name, None, None)


# Check availability at import time
def check_pdflatex_available() -> bool:
    """Check if pdflatex is available on the system."""
    _log = logging.getLogger(__name__)
    try:
        result = subprocess.run(['pdflatex', '--version'], capture_output=True, timeout=5)
        available = result.returncode == 0
        _log.info(f"pdflatex check: available={available}, returncode={result.returncode}")
        return available
    except FileNotFoundError as e:
        _log.warning(f"pdflatex not found: {e}")
        return False
    except subprocess.TimeoutExpired:
        _log.warning("pdflatex check timed out")
        return False
    except Exception as e:
        _log.warning(f"pdflatex check failed: {e}")
        return False


PDFLATEX_AVAILABLE = check_pdflatex_available()
