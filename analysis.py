### Chris Peleties July 2026
import pandas as pd
from datetime import datetime, timedelta
from haversine import haversine,Unit
import math
import numpy as np
import matplotlib.pyplot as plt
import regex as re
import obspy
import os

#constant
shotpath = r"F:\Spatial Analysis Dist Only\Shotlog.xlsx"
sac_file_folder_path = r"F:\OBS Raw Data"
obs_info_path = r"F:\Spatial Analysis Dist Only\OBS Instruments.xlsx"
output_folder = r"F:\Spatial Analysis Dist Only"
predictions_folder = r"F:\Spatial Analysis Dist Only\predictions"

upa_per_count = 48.25

def get_obs_position(obs_info_path,predictions_path):
    obs_data = pd.read_excel(obs_info_path,sheet_name="Instrument Summary")
   
    match = re.search(r"(OBH|OBS)\d+", predictions_path.upper())
    instrument = match.group(0) if match else None
    
    row = (obs_data["Station"] == instrument)
    obs_data = obs_data[row]

    
    if len(obs_data) < 1:
        print("no instrument found")
        return None, None
    else:
        lat = obs_data["Lat"].iloc[0]
        long = obs_data["Long"].iloc[0]

        return lat,long,instrument

def get_shot_datetime(year, julian_day, time_value):
   #get a datetime from the shot log columns

    year = int(year)
    julian_day = int(julian_day)

    date_at_start_of_year = datetime(year, 1, 1)
    shot_date = date_at_start_of_year + timedelta(days=julian_day - 1)

    # get time from excel
    if hasattr(time_value, "hour"):
        shot_datetime = shot_date + timedelta(
            hours=time_value.hour,
            minutes=time_value.minute,
            seconds=time_value.second,
            microseconds=time_value.microsecond
        )

    return shot_datetime


def read_shot_file(shot_path):

    shots = pd.read_excel(shot_path, header=None)

    shots = shots.iloc[:, [0,1,2,3,4,5]].copy()

    shots.columns = [
        "shot_number",
        "year",
        "julian_day",
        "shot_times",
        "latitude",
        "longitude",
    ]

    shot_times = []

    for i in range(len(shots)):
        shot_time = get_shot_datetime(shots.loc[i, "year"],shots.loc[i, "julian_day"],shots.loc[i, "shot_times"])

        shot_times.append(shot_time)

    shots["shot_time"] = shot_times

    #sorts shot times in chronological order (if they arent already)
    shots = shots.sort_values("shot_time").reset_index(drop=True)

    print("Shot file:", shot_path,"Number of shots:", len(shots),"First shot:", shots["shot_time"].min(),"Last shot:", shots["shot_time"].max())

    return shots

def get_average_ship_position(shots,start_time,end_time):

    shots_in_period = (shots["shot_time"] >= start_time) & (shots["shot_time"] < end_time)
    shots_in_period = shots[shots_in_period].copy() #filter df 

    if len(shots_in_period) > 0:
        lat = shots_in_period["latitude"].mean()
        long = shots_in_period["longitude"].mean()
    else:
        lat = None
        long = None

    return lat,long


def read_predictions_file(predictions_path):
    predictions = pd.read_csv(predictions_path)

    #correct datetimes (-30 mins from each)
    predictions["corrected_datetime"] = pd.to_datetime(predictions["file_datetime"],format="%Y-%m-%dT%H:%M:%S") - pd.Timedelta(minutes=30)
    predictions["corrected_datetime"] = predictions["corrected_datetime"] + pd.to_timedelta(predictions["clip_start_sec"],unit="s")


    return predictions


def distance_bins(results):
    
    bin_distances_km = 5
    #get max distance and round up to nearest 5
    max_dist = results["Distance (Km)"].max()
    print(max_dist)
    upper_bin_limit = math.ceil(max_dist/5)*5
    print(upper_bin_limit)
    
    #make an array of bins
    bins = np.arange(0,upper_bin_limit+bin_distances_km,bin_distances_km)
    
    results["bin"] = pd.cut(results["Distance (Km)"],bins)


    binned_results= results.groupby("bin",observed=False).agg({'Positive_frame':'sum','Total_frames':'count'}).reset_index()
    #probability of positive frame
    # Midpoint of each interval
    binned_results["bin_midpoint"] = (binned_results["bin"].apply(lambda x: x.mid))
    binned_results["prob_positive_segment"] = binned_results["Positive_frame"]/binned_results["Total_frames"]

    return binned_results    


def get_sac_from_wav_name(filename,segment_start):
    
    match = re.search(r"(OBH|OBS)\d+", filename.upper())
    instrument = match.group(0) if match else None
    
    segment_start = pd.Timestamp(segment_start)
    julian_day = segment_start.dayofyear
    
    sac_path_end = f"{instrument}.JD{julian_day:03d}.CH3.SAC"
    
    sac_path = sac_file_folder_path + '\\' + sac_path_end
    
    print(sac_path)
    return sac_path

def longest_consecutive_run(values):

    longest_run = 0
    current_run = 0

    for value in values:
        if value == 1:
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0

    return longest_run

def get_segment_pressures(current_segment_start,current_segment_end, sac_path, upa_per_count):
    #function to find pressures for the given segment using calibration data

    st = obspy.read(str(sac_path))
    tr = st[0]

    sac_start_time = tr.stats.starttime.datetime
    sampling_rate = tr.stats.sampling_rate
    dt = 1.0 / sampling_rate
    
    #find the index of the start and end of the segment in the sac file
    delta_t = timedelta(seconds=dt)
    timeshift = current_segment_start - sac_start_time
    end_timeshift = current_segment_end - sac_start_time
    
    start_sample = int(timeshift / delta_t)
    end_sample = int(end_timeshift / delta_t)
    
    segment_counts = tr.data[start_sample:end_sample].astype(float)

    pressure_uPa = segment_counts * upa_per_count
    pressure_Pa = pressure_uPa / 1e6

    print("SAC file:", sac_path,"Start time:", sac_start_time,"Segment start", current_segment_start, "Number of samples:", len(pressure_Pa))
    
    return pressure_Pa, sac_start_time, sampling_rate, dt
    
def calculate_peak_spl(pressure_frame_Pa, dt):
    #Inserts the pressure column from the sac file

    # Find peak pressure
    peak_pressure_Pa = np.max(np.abs(pressure_frame_Pa))

    #handle nulls
    if peak_pressure_Pa > 0:
        peak_SPL_dB = 20.0 * np.log10(peak_pressure_Pa / 1e-6)
    else:
        peak_SPL_dB = np.nan

    # SEL
    pressure_frame_uPa = pressure_frame_Pa * 1e6
    sound_exposure_uPa2s = np.sum(pressure_frame_uPa ** 2) * dt

    #handle nulls
    if sound_exposure_uPa2s > 0:
        SEL_dB = 10.0 * np.log10(sound_exposure_uPa2s)
    else:
        SEL_dB = np.nan

    return peak_pressure_Pa, peak_SPL_dB, SEL_dB, sound_exposure_uPa2s

def noise_level_bins(results):
    
    bin_size_dB = 5
    
    #get max SPL and lower as wont be 0 and round up to nearest 5
    max_spl = results["Peak_SPL_dB"].max()
    min_spl = results["Peak_SPL_dB"].min()
    upper_bin_limit = math.ceil(max_spl/bin_size_dB)*bin_size_dB
    lower_bin_limit = math.floor(min_spl/bin_size_dB)*bin_size_dB
    print(lower_bin_limit,"to",upper_bin_limit)
    
    #make an array of bins
    bins = np.arange(lower_bin_limit,upper_bin_limit+bin_size_dB,bin_size_dB)
    
    results["bin"] = pd.cut(results["Peak_SPL_dB"],bins)


    binned_results= results.groupby("bin",observed=False).agg({'Positive_frame':'sum','Total_frames':'count'}).reset_index()
    #probability of positive frame
    # Midpoint of each interval
    binned_results["bin_midpoint_dB"] = (binned_results["bin"].apply(lambda x: x.mid))
    binned_results["prob_positive_segment"] = binned_results["Positive_frame"]/binned_results["Total_frames"]

    return binned_results    


## Main Processing Function 

def process_5_min_segments(shots,predictions):
    results = []
    first_shot_time = pd.Timestamp(shots["shot_time"].min())
    first_segment_start = first_shot_time.round(freq='5min')
    last_segment_end = pd.Timestamp(shots["shot_time"].max()).ceil('5min')
    #first_segment_end = first_segment_start + pd.Timedelta(minutes=4,seconds=59)
    
    current_segment_start = first_segment_start
    current_segment_end = current_segment_start + pd.Timedelta(minutes =5)
    
    while current_segment_end <= last_segment_end:
        
        #get only predictions in this timeframe
        segment_predictions = []
        segment_predictions = (predictions["corrected_datetime"] >= current_segment_start) & (predictions["corrected_datetime"] < current_segment_end)
        segment_predictions = predictions[segment_predictions].copy()
        
        #check if the predictions file ends before the shooting file (if OBS stopped recording) and skip segment if it does
        if len(segment_predictions) == 0:
            print("No predictions for:", current_segment_start, "to", current_segment_end)
        
            current_segment_start += pd.Timedelta(minutes=5)
            current_segment_end += pd.Timedelta(minutes=5)
            continue
        
        #find positives
        total_frames = len(segment_predictions)
        positives = segment_predictions["prediction"].sum()
        
        longest_positive_run = longest_consecutive_run(segment_predictions["prediction"])
        
        if positives >= 5 or longest_positive_run >= 4:
            positive_frame = 1
        else:
            positive_frame = 0

        #get number of shots
        segment_shots = []
        segment_shots = (shots["shot_time"] >= current_segment_start) & (shots["shot_time"] < current_segment_end)
        segment_shots = shots[segment_shots].copy()
        number_of_shots = len(segment_shots)
        
        if number_of_shots == 0: 
            distance_m = 0 
            segment_ship_lat = None
            segment_ship_long = None
            peak_pressure_Pa = None
            peak_SPL_dB = None
            SEL_dB = None
            sound_exposure_uPa2s = None
            #append results for this frame
            results.append({"Segment_Start": current_segment_start, "Segment_End": current_segment_end,
                            "Total_frames": total_frames,"Positives": positives,"Positive_frame": positive_frame,"Longest_positive_run": longest_positive_run,
                            "Ship_lat": segment_ship_lat,"Ship_long": segment_ship_long, "Number of shots": number_of_shots,
                            "Distance (Km)":distance_m,"Peak_Pressure_Pa": peak_pressure_Pa, "Peak_SPL_dB": peak_SPL_dB,"Sound_Exposure_Level":SEL_dB, "Sound_Exposure": sound_exposure_uPa2s})
            print("Current Segment from:",current_segment_start,"to",current_segment_end," No Shooting")
            
            #move onto next segment
            current_segment_start += pd.Timedelta(minutes = 5)
            current_segment_end += pd.Timedelta(minutes = 5)
            
        #when there IS shooting    
        else:
            #get ship position
            segment_ship_lat,segment_ship_long= get_average_ship_position(shots, current_segment_start, current_segment_end)
            #get ship distance
            distance_m = haversine((OBS_LATITUDE,OBS_LONGITUDE),(segment_ship_lat,segment_ship_long))
            
            #Get the pressures
            source_audio = segment_predictions["source_wav"].min()
            sac_file = get_sac_from_wav_name(source_audio,current_segment_start)
            
            segment_pressures,sac_start_time, sampling_rate, dt = get_segment_pressures(current_segment_start, current_segment_end, sac_file, upa_per_count)
            
            if len(segment_pressures) == 0:
                print("No pressure samples for this segment, setting sound to None")
            
                peak_pressure_Pa = None
                peak_SPL_dB = None
                SEL_dB = None
                sound_exposure_uPa2s = None
            
            else:
                peak_pressure_Pa,peak_SPL_dB,SEL_dB,sound_exposure_uPa2s = calculate_peak_spl(segment_pressures, dt)
            
            
            #append all results for this frame
            results.append({"Segment_Start": current_segment_start, "Segment_End": current_segment_end,
                        "Total_frames": total_frames,"Positives": positives,"Positive_frame": positive_frame,"Longest_positive_run": longest_positive_run,
                        "Ship_lat": segment_ship_lat,"Ship_long": segment_ship_long, "Number of shots": number_of_shots,
                        "Distance (Km)":distance_m,"Peak_Pressure_Pa": peak_pressure_Pa, "Peak_SPL_dB": peak_SPL_dB,"Sound_Exposure_Level":SEL_dB,"Sound_Exposure": sound_exposure_uPa2s})
        
            
            print("Current Segment from:",current_segment_start,"to",current_segment_end,"Positives: ", positives)
            
            
            current_segment_start += pd.Timedelta(minutes = 5)
            current_segment_end += pd.Timedelta(minutes = 5)
        
    results = pd.DataFrame(results)
    
    #Shooting Totals
    shooting_results = results[results["Number of shots"] > 0].copy()
    total_shooting_segments = len(shooting_results)
    positive_shooting_segments = shooting_results["Positive_frame"].sum()
    
    if total_shooting_segments > 0:
        percent_positive_shooting_segments = (positive_shooting_segments / total_shooting_segments) * 100
    else: 
        percent_positive_shooting_segments = 0
    
    print("Total shooting 5-min segments:", total_shooting_segments)
    print("Positive shooting 5-min segments:", positive_shooting_segments)
    print("% positive shooting 5-min segments:", percent_positive_shooting_segments)
   
    
    return results

def plot_distance(all_binned_results):
    plt.figure(figsize=(10, 6))

    for instrument in all_binned_results["Instrument"].unique():

        instrument_data = all_binned_results[all_binned_results["Instrument"] == instrument]

        plt.scatter(instrument_data["bin_midpoint"],instrument_data["prob_positive_segment"],label=instrument)

    plt.xlabel("Distance from ship to OBS (Km)")
    plt.ylabel("% of positive 5 min segments")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.savefig(r"F:\Spatial Analysis Dist Only\distance_vs_positive_segments.png")
    plt.show()
    
    
def plot_noise(all_binned_sound_results):
    plt.figure(figsize=(10, 6))

    for instrument in all_binned_sound_results["Instrument"].unique():

        instrument_data = all_binned_sound_results[all_binned_sound_results["Instrument"] == instrument]

        plt.scatter(instrument_data["bin_midpoint_dB"],instrument_data["prob_positive_segment"],label=instrument)

    plt.xlabel("Peak SPL at OBS (dB re μPa)")
    plt.ylabel("% of positive 5 min segments")
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.savefig(r"F:\Spatial Analysis Dist Only\dB_vs_positive_segments.png")
    plt.show()
    
def plot_single_distance(binned_results, instrument):
    plt.figure(figsize=(10, 6))

    plt.scatter(
        binned_results["bin_midpoint"],
        binned_results["prob_positive_segment"]
    )

    plt.xlabel("Distance from ship to OBS (Km)")
    plt.ylabel("% of positive 5 min segments")
    plt.title(instrument)
    plt.ylim(0, 1)
    plt.grid(True)

    save_path = rf"F:\Spatial Analysis Dist Only\distance_vs_positive_segments_{instrument}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    
def plot_single_noise(binned_sound_results, instrument):
    plt.figure(figsize=(10, 6))

    plt.scatter(
        binned_sound_results["bin_midpoint_dB"],
        binned_sound_results["prob_positive_segment"]
    )

    plt.xlabel("Peak SPL at OBS (dB re μPa)")
    plt.ylabel("% of positive 5 min segments")
    plt.title(instrument)
    plt.ylim(0, 1)
    plt.grid(True)

    save_path = rf"F:\Spatial Analysis Dist Only\dB_vs_positive_segments_{instrument}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

#############
#Final Script
##############
all_binned_results = []
all_binned_sound_results = []


shots = read_shot_file(shotpath)
for predictions_file in os.listdir(predictions_folder):
    
    if not predictions_file.lower().endswith(".csv"):
        continue

    predictions_path = predictions_folder + "\\" + predictions_file
    OBS_LATITUDE,OBS_LONGITUDE, instrument = get_obs_position(obs_info_path, predictions_path)
    print("Instrument Being Processed: ",instrument, "Path: ", predictions_path)
    
    predictions = read_predictions_file(predictions_path)

    results = process_5_min_segments(shots,predictions)
    
    binned_results = distance_bins(results)
    #add the current instrument
    binned_results["Instrument"] = instrument
    
    binned_sound_results = noise_level_bins(results)
    binned_sound_results["Instrument"] = instrument
    
    # save this instruments plots 
    plot_single_distance(binned_results, instrument)
    plot_single_noise(binned_sound_results, instrument)
    
    #add to all_files 
    all_binned_results.append(binned_results)
    all_binned_sound_results.append(binned_sound_results)
    
    ######SAVE FILES################
    results_output_path = output_folder + "\\" + instrument + "_5min_results.csv"
    results.to_csv(results_output_path, index=False)
    
    binned_output_path = output_folder + "\\" + instrument + "_distance_binned_results.csv"
    binned_results.to_csv(binned_output_path, index=False)
    
    binned_sound_output_path = output_folder + "\\" + instrument + "_sound_binned_results.csv"
    binned_sound_results.to_csv(binned_sound_output_path, index=False)

    ##################################
#make df
all_binned_results = pd.concat(all_binned_results,ignore_index=True)
all_binned_sound_results = pd.concat(all_binned_sound_results,ignore_index=True)
    
plot_distance(all_binned_results)
plot_noise(all_binned_sound_results)
