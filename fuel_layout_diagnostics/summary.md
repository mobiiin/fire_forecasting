# Fuel Layout Diagnostics

- Loaded file: /media/mhabibp/Elements/Mobin CPS files/nosolar_actualNIROPS_asc_1of2_081214/data/KINGNSM04ASC.fuel.1918
- Total values: 1036800
- Finite values: 1036800
- NaNs: 0
- Zeros: 249154
- Negative values: 0
- Min: 0
- Max: 7.749
- Mean: 1.46986
- Std: 1.09215
- First 20 numeric values: [0.896, 0.784, 1.12, 0.896, 0.784, 0.896, 0.896, 0.896, 2.692, 0.896, 1.344, 0.896, 2.692, 0.784, 0.78, 0.0, 0.784, 3.591, 0.784, 0.784]

## Parsed Header

- raw_first_line: 720   720    2    95.023    74.074
- raw_first_line_numbers: [720.0, 720.0, 2.0, 95.023, 74.074]
- ncols_guess: 720
- nrows_guess: 720
- channels_guess: 2
- extra_first_line_numbers: [95.023, 74.074]

## Candidate Shapes

- 720 x 720 | source=half:518400 | score=0.0000

## Generated Figures

- layoutB_channel_major_720x720_H720_W720_orderC.png
- layoutB_channel_major_720x720_H720_W720_orderF.png
- layoutC_interleaved_720x720_H720_W720_orderC.png
- layoutC_interleaved_720x720_H720_W720_orderF.png
- layoutD_vhw_720x720_orderC.png
- layoutE_hwv_720x720_orderC.png

## Tile Correlation Notes

- layoutB_channel_major_720x720: surface tile scores q1_q2=-0.0857, q1_q3=-0.0195, q1_q4=0.0498
- layoutB_channel_major_720x720: canopy tile scores q1_q2=-0.0735, q1_q3=-0.0444, q1_q4=0.0289
- layoutB_channel_major_720x720: surface tile scores q1_q2=-0.0195, q1_q3=-0.0857, q1_q4=0.0498
- layoutB_channel_major_720x720: canopy tile scores q1_q2=-0.0444, q1_q3=-0.0735, q1_q4=0.0289
- layoutC_interleaved_720x720: surface tile scores q1_q2=0.6666, q1_q3=0.6211, q1_q4=0.5087
- layoutC_interleaved_720x720: canopy tile scores q1_q2=0.6698, q1_q3=0.6219, q1_q4=0.5107
- layoutC_interleaved_720x720: surface tile scores q1_q2=0.6211, q1_q3=0.6666, q1_q4=0.5087
- layoutC_interleaved_720x720: canopy tile scores q1_q2=0.6219, q1_q3=0.6698, q1_q4=0.5107
- layoutD_720x720: surface tile scores q1_q2=-0.0857, q1_q3=-0.0195, q1_q4=0.0498
- layoutD_720x720: canopy tile scores q1_q2=-0.0735, q1_q3=-0.0444, q1_q4=0.0289
- layoutE_720x720: surface tile scores q1_q2=0.6666, q1_q3=0.6211, q1_q4=0.5087
- layoutE_720x720: canopy tile scores q1_q2=0.6698, q1_q3=0.6219, q1_q4=0.5107

## Suspicious Layouts

- Some candidate layouts show high quadrant correlation, which is consistent with tiled or repeated maps.

## How To Read The Result

- Open the output directory and compare the PNGs side by side.
- The correct unpacking should look like one coherent top-down fuel map, not repeated quadrants, stripes, or scrambled noise.
- If a layout shows repeated 2x2 quadrants with high correlation, it is probably not the correct reshape.
