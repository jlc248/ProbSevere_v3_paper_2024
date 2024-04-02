# ProbSevere v3 paper 2024
Scripts to train and evaluate gradient-boosted decision trees like those in ProbSevere v3. 

## Install Mamba
Follow the [instructions](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html) to install `mamba`. You can also use `conda`, but I've found `mamba` tends to be faster and more robust.

## Dowload datasets
These datasets contain storm attributes for each storm at each scan/observation time. Each storm has a unique ID.
```
# Training data
wget "https://zenodo.org/records/10891478/files/storm_atts_2018-2021.pkl?download=1" -O storm_atts_2018-2021.pkl

# Validation data
wget "https://zenodo.org/records/10891478/files/storm_atts_2022.pkl?download=1" -O storm_atts_2022.pkl

# Test data
wget "https://zenodo.org/records/10891478/files/storm_atts_2023.pkl?download=1" -O storm_atts_2023.pkl
```

## Clone this project
`git clone git@github.com:jlc248/ProbSevere_v3_paper_2024.git`

## Create and activate the environment
```
cd ProbSevere_v3_paper_2024
mamba env create -f PSdemo.yml
.
.
.
mamba activate PSdemo
```

## Train a model
Before training a model, first familiarize yourself with the dataset. There is a column for `IDs`, datetimes (`dts`), `mean_lat`, and `mean_lon` of the storm. There are also columns for hail (`hail`), wind (`wind`), and tornado (`torn`) reports. The rest of the columns in the DataFrames are potential predictors/features to use.

### Column key
```
IDs: unique storm track identifier
dts: datetimes for each sample
hail: maximum reported hail size for the storm track
wind: maximum reported wind speed for the storm strack (+0.1 for estimated gusts)
torn: maximum EF scale of reported tornadoes for the storm track
first_hail_time: time of first hail report
first_wind_time: time of first wind report
first_torn_time: time of first tornado report
first_rep_time: time of first report
mean_lat: latitude centroid for the storm object
mean_lon: longitude centroid for the storm object
prob_severe: ProbSevere v1 probability
probhail: ProbSevere v2 ProbHail probability
probwind: ProbSevere v2 ProbWind probability
probtor: ProbSevere v2 ProbTor probability
max_mesh: Maximum MRMS MESH [mm]
max_compref: Maximum MRMS merged reflectivity [dBZ]
max_vil_density: Maximum MRMS VIL density [kg / m^3]
max_vil: maximum MRMS VIL [kg / m^2]
max_posh: Maximum MRMS Probability of Severe Hail []
max_shi: Maximum MRMS Severe Hail Index []
max_etop30: Maximum MRMS 30-dbZ echo top [km]
max_etop50: Maximum MRMS 50-dbZ echo top [km]
max_h50a0c: Maximum MRMS height of 50-dBZ above 0C [km]
max_h50am20c: Maximum MRMS height of 50-dBZ above -20C [km]
max_h60a0c: Maximum MRMS height of 60-dBZ above 0C [km]
max_60am20c: Maximum MRMS height of 60-dBZ above -20C [km]
max_vii: Maximum MRMS VII [kg / m^2]
max_ref10: Maximum MRMS reflectivity at -10C [dBZ]
max_ref20: Maximum MRMS reflectivity at -20C [dBZ]
max_llazshear: Maximum MRMS azimuthal shear 0-2 km AGL [ / 0.001 s]
p98_llazshear: 98th percentile MRMS azimuthal shear 0-2 km AGL [ / 0.001 s]
max_mlazshear: Maximum MRMS azimuthal shear 3-6 km AGL [ / 0.001 s]
p98_mlazshear: 98th percentile MRMS azimuthal shear 3-6 km AGL [ / 0.001 s]
flash_rate: ENTLN flash rate (i.e., 2-min sum of flash density over storm) [flashes]
max_tltg_density: Maximum ENTLN flash density [flashes / min / km^2]
dfrdt: Change in ENTLN flash rate [flashes / min]
lja_std: Lightning Jump Algorithm (Schultz et al. 2011) standard deviation / sigma []
mucape: RAP most-unstable CAPE [J / kg]
mlcape: RAP 0-90 mb AGL mixed-layer CAPE [J / kg]
cape_M10M30: RAP CAPE between -10C and -30C ("hail CAPE") [J / kg]
mlcin: RAP 0-90 mb mixed-layer CIN [J / kg]
wbz: RAP lowest level of the wet-bulb 0C height [m]
pwat: RAP precipitable water [in]
ebshear: RAP effective bulk shear [kt]
meanwind_1-3kmAGL: RAP mean wind 1-3 km AGL [kt]
srh01km: RAP storm-relative helicity 0-1 km AGL [m^2 / s^2]
EBS_merged_smoothed: HRRR effective bulk shear [kt]
MLCAPE_merged_smoothed: HRRR 0-90 mb AGL mixed-layer CAPE [J / kg]
MUCAPE_merged_smoothed: HRRR most-unstable CAPE [J / kg]
DCAPE_merged_smoothed: HRRR downdraft CAPE [J / kg]
MLCIN_merged_smoothed: HRRR 0-90 mb mixed-layer CIN [J / kg]
CAPE_M10M30_merged_smoothed: HRRR CAPE between -10C and -30C ("hail CAPE") [J / kg]
LAPSERATE_03KM_merged_smoothed: HRRR lapse rate 0-3 km AGL [C / km]
MAX_LAPSERATE_26KM_merged_smoothed: HRRR max. 2-km lapse rate between 2-6 km AGL [C / km]
SFC_LCL_merged_smoothed: HRRR surface lifted condensation level [m]
WETBULB_0C_HGT_merged_smoothed: HRRR lowest level of the wet-bulb 0C height [m]
SRH_01KM_merged_smoothed: HRRR storm-relative helicity 0-1 km AGL [m^2 / s^2]
SRW02KM_merged_smoothed: HRRR storm-relative wind 0-2 km AGL [kt]
SRW46KM_merged_smoothed: HRRR storm-relative wind 4-6 km AGL [kt]
PWAT_merged_smoothed: HRRR precipitable water [in]
MEANWIND_1-3kmAGL_merged_smoothed: HRRR mean wind 1-3 km AGL [kt]
SHEAR06KM_merged_smoothed: HRRR bulk shear 0-6 km AGL [kt]
TDD02KM_merged_smoothed: HRRR dewpoint depression 0-2 km AGL [C]
icp: GOES-16 IntenseStormNet (Cintineo et al. 2020) intense convection probability []
maxrc_emiss: GOES-16 normalized satellite growth rate (Cintineo et al. 2013, 2014, 2020) [% / min]
maxrc_icecf: GOES-16 change in ice-cloud-fraction (Cintineo et al. 2013, 2014) [ / min]
popdens: Population density from 2015 dataset [persons / km^2]
size: size of the storm, roughly equivalent to area in km^2 [number of pixels]
beam_hgt: The radar beam height AGL, assuming a standard atmosphere [m]
max_fed: Maximum GLM flash-extent density [flashes / 5-min]
max_fcd: Maximum GLM flash-centroid density [flashes / 5-min]
accum_fcd: Sum of GLM flash-centroid density ~GLM flash rate [flashes]
max_total_energy: Maximum GLM total optical energy [fJ ?]
accum_total_energy: Sum of GLM total optical energy [fJ ?]
avg_flash_area: Average GLM flash area [km^2]
avg_group_area: Average GLM group area [km^2]
severe_climo: Any-severe hazard climatology as a function of location and day-of-year []
hail_climo: Severe hail climatology as a function of location and day-of-year []
wind_climo: Severe wind climatology as a function of location and day-of-year []
tor_climo: Tornado climatology as a function of location and day-of-year []
elev: Elevation above sea level [m]
```
Use the list above and your knowledge of severe weather to modify the predictor lists in `train_LGBM_mode.py`. These lists are approximately on lines 125-150, with different blocks for the different hazard-type models. Once you have your predictor list configured, you can start training a model.

### Help message

```
(PSdemo) [user@machine ProbSevere_v3_paper_2024]$ python train_LGBM_model.py -h
usage: train_LGBM_model.py [-h] [-o OUTDIR] [-pt] [-ph] [-pw] [-bp] [-es EARLY_STOPPING_ROUNDS] train val

Train and validate a LGBM GBDT. Default is to train against 'any severe' report.

positional arguments:
  train                 Training DataFrame
  val                   Validation DataFrame

options:
  -h, --help            show this help message and exit
  -o OUTDIR, --outdir OUTDIR
                        Root output directory for model and figures. Default=$PWD/OUTPUT/
  -pt, --probtor_test   Train and test a tornado only model
  -ph, --probhail_test  Train and test a hail only model
  -pw, --probwind_test  Train and test a wind only model
  -bp, --best_params    Train a model with pre-defined optimal parameters (hard-coded in function). Use in conjuction with -pt, -ph, -pw or none (which
                        would be prob any severe)
  -es EARLY_STOPPING_ROUNDS, --early_stopping_rounds EARLY_STOPPING_ROUNDS
                        Number of early stopping rounds. Default = 5. Make this a huge number if you don't want early stopping.
```

The program requires a training and validation DataFrame. You can use the ones you downloaded. `--best_params` will use the pre-defined parameters from the hyperparameter optimization in Cintineo et al. (2024). The program defaults to using 40 CPU threads, and will take a few minutes to run (< 10 min). However, you can change the number of CPUs if you'd like (line 210 `params['num_threads'] = 40`).

### Run the training

This will train a model to predict severe hail. If neither `-pt`, `-ph`, nor `-pw` are selected, the default will be to train a "all/any hazards" model.

`python train_LGBM_model.py storm_atts_2018-2021.pkl storm_atts_2022.pkl -o test01/ --best_params -ph -es 5`

The output files are:
-  `lgb_classifier.pkl` (the model)
-  `verification_scores.txt` (basic evaluation stats on the validation set)
-  `params.pkl`
-  `scores.pkl` (contains CSI, POD, success ratio, bias, accuracy, PSS)
-  `val_lab_pred.pkl` (contains the validation labels and final predictions)
-  `attributes_diagram.png`
-  `performance_diagram.png`
    -  It should be noted that CSI here is: `CSI = hits / (hits + misses + false_alarms)`, which is different than the CSI computation in the paper.
-  `feature_importance_gain.png`
-  `feature_importance_split.png`

<img src="https://github.com/jlc248/ProbSevere_v3_paper_2024/assets/19976290/2c8ab40c-8a35-4f0c-beec-82e9535212c9" width=40%>

<img src="https://github.com/jlc248/ProbSevere_v3_paper_2024/assets/19976290/5d74fe92-db90-46fb-b9a4-590162292ccf" width=40%>

<img src="https://github.com/jlc248/ProbSevere_v3_paper_2024/assets/19976290/bbb09706-9f5c-420c-a15b-3d7660fe41c9" width=75%>




## Citation
