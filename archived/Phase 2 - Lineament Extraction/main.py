import datetime
from helpers import *

def main():
	# Hillshading
	hillshade_start = datetime.datetime.now()
	hillshade(src_dir='raw_1m', in_path="/media/riza/'Seagate Bac'/NEW/lineament_extraction/raw_1m/", out_path="/media/riza/'Seagate Bac'/NEW/lineament_extraction/hillshaded_1m/")
	hillshade(src_dir='raw_5m', in_path="/media/riza/'Seagate Bac'/NEW/lineament_extraction/raw_5m/", out_path="/media/riza/'Seagate Bac'/NEW/lineament_extraction/hillshaded_5m/")
	hillshade(src_dir='raw_10m', in_path="/media/riza/'Seagate Bac'/NEW/lineament_extraction/raw_10m/", out_path="/media/riza/'Seagate Bac'/NEW/lineament_extraction/hillshaded_10m/")
	hillshade_time = datetime.datetime.now() - hillshade_start
	print('HILLSHADE DONE: ', hillshade_time)


	# Lineament extraction experiments
	# 1: 1M resolution, no smoothing, canny edge detection
	exp1_start = datetime.datetime.now()
	exp1_line_start = datetime.datetime.now()
	extract_lineaments(src_base='hillshaded_1m', dest_base='results/exp1', smoothing=False, edge_method='canny')
	for_plotting(csv_file='results/exp1/lineaments_data.csv', dest_dir='results/points_data', dest_file='lineaments1_points.csv')
	exp1_line_time = datetime.datetime.now() - exp1_line_start
	print('Exp1 extraction done: ', exp1_line_time)

	exp1_map_start = datetime.datetime.now()
	map_lineaments(csv_file='results/exp1/lineaments_data.csv', src_dir='hillshaded_1m/multi', dest_dir='results/exp1/compiled')
	exp1_map_time = datetime.datetime.now() - exp1_map_start
	print('Exp1 mapping done: ', exp1_map_time)

	exp1_time = datetime.datetime.now() - exp1_start
	print('EXP1 DONE: ', exp1_time)


	# 2: 1M resolution, with smoothing, canny edge detection
	exp2_start = datetime.datetime.now()
	exp2_line_start = datetime.datetime.now()
	extract_lineaments(src_base='hillshaded_1m', dest_base='results/exp2', smoothing=True, edge_method='canny')
	for_plotting(csv_file='results/exp2/lineaments_data.csv', dest_dir='results/points_data', dest_file='lineaments2_points.csv')
	exp2_line_time = datetime.datetime.now() - exp2_line_start
	print('Exp2 extraction done: ', exp2_line_time)

	exp2_map_start = datetime.datetime.now()
	map_lineaments(csv_file='results/exp2/lineaments_data.csv', src_dir='hillshaded_1m/multi', dest_dir='results/exp2/compiled')
	exp2_map_time = datetime.datetime.now() - exp2_map_start
	print('Exp2 mapping done: ', exp2_map_time)

	exp2_time = datetime.datetime.now() - exp2_start
	print('EXP2 DONE: ', exp2_time)


	# 3: 5M resolution, with smoothing, canny edge detection
	exp3_start = datetime.datetime.now()
	exp3_line_start = datetime.datetime.now()
	extract_lineaments(src_base='hillshaded_5m', dest_base='results/exp3', smoothing=True, edge_method='canny')
	for_plotting(csv_file='results/exp3/lineaments_data.csv', dest_dir='results/points_data', dest_file='lineaments3_points.csv')
	exp3_line_time = datetime.datetime.now() - exp3_line_start
	print('Exp3 extraction done: ', exp3_line_time)

	exp3_map_start = datetime.datetime.now()
	map_lineaments(csv_file='results/exp3/lineaments_data.csv', src_dir='hillshaded_5m/multi', dest_dir='results/exp3/compiled')
	exp3_map_time = datetime.datetime.now() - exp3_map_start
	print('Exp3 mapping done: ', exp3_map_time)

	exp3_time = datetime.datetime.now() - exp3_start
	print('EXP3 DONE: ', exp3_time)
	

	# 4: 10M resolution, with smoothing, canny edge detection
	exp4_start = datetime.datetime.now()
	exp4_line_start = datetime.datetime.now()
	extract_lineaments(src_base='hillshaded_10m', dest_base='results/exp4', smoothing=True, edge_method='canny')
	for_plotting(csv_file='results/exp4/lineaments_data.csv', dest_dir='results/points_data', dest_file='lineaments4_points.csv')
	exp4_line_time = datetime.datetime.now() - exp4_line_start
	print('Exp4 extraction done: ', exp4_line_time)

	exp4_map_start = datetime.datetime.now()
	map_lineaments(csv_file='results/exp4/lineaments_data.csv', src_dir='hillshaded_10m/multi', dest_dir='results/exp4/compiled')
	exp4_map_time = datetime.datetime.now() - exp4_map_start
	print('Exp4 mapping done: ', exp4_map_time)

	exp4_time = datetime.datetime.now() - exp4_start
	print('EXP4 DONE: ', exp4_time)


	# 5: 1M resolution, no smoothing, phase congruency edge detection
	exp5_start = datetime.datetime.now()
	exp5_line_start = datetime.datetime.now()
	extract_lineaments(src_base='hillshaded_1m', dest_base='results/exp5', smoothing=False, edge_method='phase_cong')
	for_plotting(csv_file='results/exp5/lineaments_data.csv', dest_dir='results/points_data', dest_file='lineaments5_points.csv')
	exp5_line_time = datetime.datetime.now() - exp5_line_start
	print('Exp5 extraction done: ', exp5_line_time)

	exp5_map_start = datetime.datetime.now()
	map_lineaments(csv_file='results/exp5/lineaments_data.csv', src_dir='hillshaded_1m/multi', dest_dir='results/exp5/compiled')
	exp5_map_time = datetime.datetime.now() - exp5_map_start
	print('Exp5 mapping done: ', exp5_map_time)

	exp5_time = datetime.datetime.now() - exp5_start
	print('EXP5 DONE: ', exp5_time)


	# 6: 1M resolution, no smoothing, phase symmetry edge detection
	exp6_start = datetime.datetime.now()
	exp6_line_start = datetime.datetime.now()
	extract_lineaments(src_base='hillshaded_1m', dest_base='results/exp6', smoothing=False, edge_method='phase_sym')
	for_plotting(csv_file='results/exp6/lineaments_data.csv', dest_dir='results/points_data', dest_file='lineaments6_points.csv')
	exp6_line_time = datetime.datetime.now() - exp6_line_start
	print('Exp6 extraction done: ', exp6_line_time)

	exp6_map_start = datetime.datetime.now()
	map_lineaments(csv_file='results/exp6/lineaments_data.csv', src_dir='hillshaded_1m/multi', dest_dir='results/exp6/compiled')
	exp6_map_time = datetime.datetime.now() - exp6_map_start
	print('Exp6 mapping done: ', exp6_map_time)

	exp6_time = datetime.datetime.now() - exp6_start
	print('EXP6 DONE: ', exp6_time)


	# 7: 1M resolution, with smoothing, phase congruency edge detection
	exp7_start = datetime.datetime.now()
	exp7_line_start = datetime.datetime.now()
	extract_lineaments(src_base='hillshaded_1m', dest_base='results/exp7', smoothing=True, edge_method='phase_cong')
	for_plotting(csv_file='results/exp7/lineaments_data.csv', dest_dir='results/points_data', dest_file='lineaments7_points.csv')
	exp7_line_time = datetime.datetime.now() - exp7_line_start
	print('Exp7 extraction done: ', exp7_line_time)

	exp7_map_start = datetime.datetime.now()
	map_lineaments(csv_file='results/exp7/lineaments_data.csv', src_dir='hillshaded_1m/multi', dest_dir='results/exp7/compiled')
	exp7_map_time = datetime.datetime.now() - exp7_map_start
	print('Exp7 mapping done: ', exp7_map_time)

	exp7_time = datetime.datetime.now() - exp7_start
	print('EXP7 DONE: ', exp7_time)


	# 8: 5M resolution, with smoothing, phase congruency edge detection
	exp8_start = datetime.datetime.now()
	exp8_line_start = datetime.datetime.now()
	extract_lineaments(src_base='hillshaded_5m', dest_base='results/exp8', smoothing=True, edge_method='phase_cong')
	for_plotting(csv_file='results/exp8/lineaments_data.csv', dest_dir='results/points_data', dest_file='lineaments8_points.csv')
	exp8_line_time = datetime.datetime.now() - exp8_line_start
	print('Exp8 extraction done: ', exp8_line_time)

	exp8_map_start = datetime.datetime.now()
	map_lineaments(csv_file='results/exp8/lineaments_data.csv', src_dir='hillshaded_5m/multi', dest_dir='results/exp8/compiled')
	exp8_map_time = datetime.datetime.now() - exp8_map_start
	print('Exp8 mapping done: ', exp8_map_time)

	exp8_time = datetime.datetime.now() - exp8_start
	print('EXP8 DONE: ', exp8_time)


	# 9: 10M resolution, with smoothing, phase congruency edge detection
	exp9_start = datetime.datetime.now()
	exp9_line_start = datetime.datetime.now()
	extract_lineaments(src_base='hillshaded_10m', dest_base='results/exp9', smoothing=True, edge_method='phase_cong')
	for_plotting(csv_file='results/exp9/lineaments_data.csv', dest_dir='results/points_data', dest_file='lineaments9_points.csv')
	exp9_line_time = datetime.datetime.now() - exp9_line_start
	print('Exp9 extraction done: ', exp9_line_time)

	exp9_map_start = datetime.datetime.now()
	map_lineaments(csv_file='results/exp9/lineaments_data.csv', src_dir='hillshaded_10m/multi', dest_dir='results/exp9/compiled')
	exp9_map_time = datetime.datetime.now() - exp9_map_start
	print('Exp9 mapping done: ', exp9_map_time)

	exp9_time = datetime.datetime.now() - exp9_start
	print('EXP9 DONE: ', exp9_time)

if __name__=="__main__": 
	main()