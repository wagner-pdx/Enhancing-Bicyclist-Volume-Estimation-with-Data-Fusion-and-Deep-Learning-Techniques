key_features = \
[
    "site_id",
]

monthly_features = \
[
    "month",
]

daily_features = \
[
    'month',                              # from combined-pc-counts-daily-stv20XX
    "day",
    'weekday',                            # from combined-pc-counts-daily-stv20XX
]

#
#  Strava Features
#     stv_adb == stv_c_adb + stv_nc_adb
#
daily_strava_features = \
[
    'Counts',                             # from combined-pc-counts-daily-stv20XX
    'stv_daily',                          # from combined-pc-counts-daily-stv20XX
    'stv_c_daily',                        # from combined-pc-counts-daily-stv20XX
    'n_links',                            # from combined-pc-counts-daily-stv20XX
]

monthly_strava_features = \
[
    "Counts",                             # from combined-pc-counts-monthly-stv20XX
    "stv_monthly",                        # from combined-pc-counts-monthly-stv20XX
    "stv_c_monthly",                      # from combined-pc-counts-monthly-stv20XX
    "n_links",                            # from combined-pc-counts-monthly-stv20XX
]

yearly_strava_features = \
[
    "stv_adb",                            # from df1-20XX
    "stv_c_adb",                          # from df1-20XX
    "stv_nc_adb",                         # from df1-20XX
    "log_stv_adb",                        # from df1-20XX
    "log_stv_c_adb",                      # from df1-20XX
    "log_stv_nc_adb",                     # from df1-20XX
]

#
#  Target Features
#
monthly_target_features = \
[
    "madb",                               # from combined-pc-counts-monthly-stv20XX
]

yearly_target_features = \
[
    "adb",                                # from df1-20XX
    "aadb1",                              # from df1-20XX
    "aadb2",                              # from df1-20XX
    "aadb",                               # from df1-20XX
    "AADBT",                              # from df1-20XX
    "reg_num",                            # from df1-20XX
]

#
#  Static Features
#
static_features = \
[
    "primary_binary",                     # from df1-20XX
    "secondary_binary",                   # from df1-20XX
    "tertiary_binary",                    # from df1-20XX
    "residential_binary",                 # from df1-20XX
    "path_binary",                        # from df1-20XX
    "cycleway_binary",                    # from df1-20XX
    "footway_binary",                     # from df1-20XX
    "cycleway_lane_binary",               # from df1-20XX
    "cycleway_track_all_binary",          # from df1-20XX
    "min_dist_to_city",                   # from df1-20XX
    "min_dist_to_park",                   # from df1-20XX
    "min_dist_to_polygon",                # from df1-20XX
    "min_dist_to_school",                 # from df1-20XX
    "min_dist_to_college",                # from df1-20XX
    "min_dist_to_university",             # from df1-20XX
    "min_dist_to_maj_uni",                # from df1-20XX
    "lanes_hm",                           # from df1-20XX
    "maxspeed_hm",                        # from df1-20XX
    "Intersection_Density_hm",            # from df1-20XX
    "Commercial Area_hm",                 # from df1-20XX
    "Distance to Commercial Area",        # from df1-20XX
    "Distance to Commercial Area Center", # from df1-20XX
    "Industrial Area_hm",                 # from df1-20XX
    "Distance to Industrial Area",        # from df1-20XX
    "Distance to Industrial Center",      # from df1-20XX
    "Residential_Area_hm",                # from df1-20XX
    "Distance to Residential Area",       # from df1-20XX
    "Distance to Residential Center",     # from df1-20XX
    "Retail Area_hm",                     # from df1-20XX
    "Distance to Retail Area",            # from df1-20XX
    "Distance to Retail Center",          # from df1-20XX
    "Grass Area_hm",                      # from df1-20XX
    "Distance to Grass",                  # from df1-20XX
    "Distance to Grass Center",           # from df1-20XX
    "Park Area_hm",                       # from df1-20XX
    "Distance to Park",                   # from df1-20XX
    "Distance to Park Center",            # from df1-20XX
    "Water Area_hm",                      # from df1-20XX
    "Distance to Water Body",             # from df1-20XX
    "Distance to Water Center",           # from df1-20XX
    "Forest Area_hm",                     # from df1-20XX
    "Distance to forest",                 # from df1-20XX
    "Distance to Forest Center",          # from df1-20XX
    "Bicycle Parking_hm",                 # from df1-20XX
    "Bus Stops_hm",                       # from df1-20XX
    "School_hm",                          # from df1-20XX
    "college_hm",                         # from df1-20XX
    "uni_count_hm",                       # from df1-20XX
    "University_hm",                      # from df1-20XX
    "maj_uni_count_hm",                   # from df1-20XX
    "Maj University_hm",                  # from df1-20XX
    "Primary_hm",                         # from df1-20XX
    "Secondary_hm",                       # from df1-20XX
    "Tertiary_hm",                        # from df1-20XX
    "Residential_Road_hm",                # from df1-20XX
    "Path_hm",                            # from df1-20XX
    "Cycleway_hm",                        # from df1-20XX
    "Footway_hm",                         # from df1-20XX
    "cycleway_lane_all_hm",               # from df1-20XX
    "cycleway_track_all_hm",              # from df1-20XX
    "Point Speed_hm",                     # from df1-20XX
    "bridge_hm",                          # from df1-20XX
    "Point Bridge_hm",                    # from df1-20XX
    "pct_white_hm",                       # from df1-20XX
    "pct_African_American_hm",            # from df1-20XX
    "pct_male_hm",                        # from df1-20XX
    "pct_female_hm",                      # from df1-20XX
    "Student Access_hm",                  # from df1-20XX
    "pct_at_least_college_education_hm",  # from df1-20XX
    "Median Age_hm",                      # from df1-20XX
    "Median_HH_income_hm",                # from df1-20XX
    "HH_density_hm",                      # from df1-20XX
    "population_density_hm",              # from df1-20XX
    "employment_density_hm",              # from df1-20XX
    "Number of jobs_hm",                  # from df1-20XX
    "point_slope_hm",                     # from df1-20XX
    "avg_slope_hm",                       # from df1-20XX
    "lanes_om",                           # from df1-20XX
    "maxspeed_om",                        # from df1-20XX
    "Intersection_Density_om",            # from df1-20XX
    "Commercial Area_om",                 # from df1-20XX
    "Industrial Area_om",                 # from df1-20XX
    "Residential_Area_om",                # from df1-20XX
    "Retail Area_om",                     # from df1-20XX
    "Grass Area_om",                      # from df1-20XX
    "Park Area_om",                       # from df1-20XX
    "Water Area_om",                      # from df1-20XX
    "Forest Area_om",                     # from df1-20XX
    "Bicycle Parking_om",                 # from df1-20XX
    "Bus Stops_om",                       # from df1-20XX
    "School_om",                          # from df1-20XX
    "college_om",                         # from df1-20XX
    "uni_count_om",                       # from df1-20XX
    "University_om",                      # from df1-20XX
    "maj_uni_count_om",                   # from df1-20XX
    "Maj University_om",                  # from df1-20XX
    "Primary_om",                         # from df1-20XX
    "Secondary_om",                       # from df1-20XX
    "Tertiary_om",                        # from df1-20XX
    "Residential_Road_om",                # from df1-20XX
    "Path_om",                            # from df1-20XX
    "Cycleway_om",                        # from df1-20XX
    "Footway_om",                         # from df1-20XX
    "cycleway_lane_all_om",               # from df1-20XX
    "cycleway_track_all_om",              # from df1-20XX
    "Point Speed_om",                     # from df1-20XX
    "bridge_om",                          # from df1-20XX
    "Point Bridge_om",                    # from df1-20XX
    "pct_white_om",                       # from df1-20XX
    "pct_African_American_om",            # from df1-20XX
    "pct_male_om",                        # from df1-20XX
    "pct_female_om",                      # from df1-20XX
    "Student Access_om",                  # from df1-20XX
    "pct_at_least_college_education_om",  # from df1-20XX
    "Median Age_om",                      # from df1-20XX
    "Median_HH_income_om",                # from df1-20XX
    "HH_density_om",                      # from df1-20XX
    "population_density_om",              # from df1-20XX
    "employment_density_om",              # from df1-20XX
    "Number of jobs_om",                  # from df1-20XX
    "point_slope_om",                     # from df1-20XX
    "avg_slope_om",                       # from df1-20XX
    "Bike_Commuter_hm",                   # from df1-20XX
    "Bike_Commuter_om",                   # from df1-20XX
    "sep_bikeway_binary",                 # from df1-20XX
    "sep_onstreet_binary",                # from df1-20XX
    "BikeFac_binary",                     # from df1-20XX
    "sep_bikeway_hm",                     # from df1-20XX
    "sep_bikeway_om",                     # from df1-20XX
    "arterial_binary",                    # from df1-20XX
    "arterial_no_bike_lane_binary",       # from df1-20XX
    "primary_no_bike_lane_binary",        # from df1-20XX
    "secondary_no_bike_lane_binary",      # from df1-20XX
    "tertiary_no_bike_lane_binary",       # from df1-20XX
    "primary_bike_lane_binary",           # from df1-20XX
    "secondary_bike_lane_binary",         # from df1-20XX
    "tertiary_bike_lane_binary",          # from df1-20XX
    "BikeFac_hm",                         # from df1-20XX
    "BikeFac_om",                         # from df1-20XX
    "BikeFac_onstreet_hm",                # from df1-20XX
    "BikeFac_onstreet_om",                # from df1-20XX
    "school_college_hm",                  # from df1-20XX
    "Park_acres_om",                      # from df1-20XX
    "Park_acres_hm",                      # from df1-20XX
    "min_dist_to_city_mi",                # from df1-20XX
    "Distance_to_Water_Body_mi",          # from df1-20XX
    "region",                             # from df1-20XX
]

#
#  Unnecessary Variable for Model-Input
#
daily_unnecessary_features = \
[
    'year',                               # from combined-pc-counts-daily-stv20XX
    'dayofweek',                          # from combined-pc-counts-daily-stv20XX
    'hourly_flg',                         # from combined-pc-counts-daily-stv20XX
    'zero_flg',                           # from combined-pc-counts-daily-stv20XX
    'nonzero_flg',                        # from combined-pc-counts-daily-stv20XX
    'daily_flg',                          # from combined-pc-counts-daily-stv20XX
    'adb_flg1',                           # from combined-pc-counts-daily-stv20XX
    'adb_flg2',                           # from combined-pc-counts-daily-stv20XX
    'Hourly_Counts_N',                    # from combined-pc-counts-daily-stv20XX
    'valid_day',                          # from combined-pc-counts-daily-stv20XX
    'review_flg',                         # from combined-pc-counts-daily-stv20XX
]

month_unnecessary_features = \
[
    "site_name_x",                        # from combined-pc-counts-monthly-stv20XX
    "year_x",                             # from combined-pc-counts-monthly-stv20XX
    "valid_days",                         # from combined-pc-counts-monthly-stv20XX
    "valid_dayofweek",                    # from combined-pc-counts-monthly-stv20XX
    "valid_month",                        # from combined-pc-counts-monthly-stv20XX
]

yearly_unnecessary_features = \
[
    "site_name.x",                        # from df1-20XX
    "year_y",                             # from df1-20XX
    "valid_months",                       # from df1-20XX
    "valid_days_year",                    # from df1-20XX
    "valid_year",                         # from df1-20XX
    "site_name.y",                        # from df1-20XX
    "edgeUID",                            # from df1-20XX
    "_osmId",                             # from df1-20XX
    "site_name_y",                        # from df1-20XX
]
