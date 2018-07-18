import pandas as pd
import numpy as np
import argparse
import json
import os
import sys
import time

parser = argparse.ArgumentParser(description="Batch process experiments.")
parser.add_argument('experiment_file', action="store", help="Path to a plaintext file with experiment list.")
parser.add_argument('ideaa_file', action="store", help="Path to ideaa file you want to process.")
parser.add_argument('ideaa_folder', action="store", help="Path to ideaa installation to find inverse files.")

crossover_data = {}
mgf_crossover_list = {}
mgf_experiment_list = {}
experiment_sums = {}
experiment_intensities = {}
norm_experiment_sums = {}
summary_files = {}
norm_total_intensities = {}
output_files = {}
ms1_index = 0
sum_index = 0


def main():
	total_start = time.time()
	arguments = parser.parse_args()

	if not os.path.isfile(arguments.experiment_file) or \
		not os.path.isfile(arguments.ideaa_file) or \
		not os.path.isdir(arguments.ideaa_folder):
		print "Please check that the paths to the experiment file, IDEAA file and IDEAA folder are correct."
		return

	print "Reading IDEAA file",
	start_time = time.time()

	ideaa_filename = arguments.ideaa_file
	ideaa_file = pd.read_csv(ideaa_filename, sep="\t")
	ideaa_file_indices = list(ideaa_file)
	sum_index = ideaa_file_indices.index("replicate_spec_flag")
	ms1_index = ideaa_file_indices.index("MS1_intensity")

	print_end_time(start_time)

	print "Reading experiments",
	start_time = time.time()
	
	sum_headers = ideaa_file.columns.values[ms1_index+1:sum_index-1]
	digest_experiments(arguments.experiment_file, arguments.ideaa_folder, sum_headers)

	print_end_time(start_time)

	print "Digesting summary",
	start_time = time.time()

	read_intensities_from_summary_and_normalize()

	print_end_time(start_time)

	#only use the columns we need: drop everything after reporter 
	#col5 = start of reporter data
	reporter_data = ideaa_file.iloc[0:,0:sum_index]

	print "Undoing previous normalization",
	start_time = time.time()
	#undo the original normalization by multiplying each reporter value by the sum
	reporter_data.iloc[0:,ms1_index+1:sum_index-1] = reporter_data.iloc[0:,ms1_index+1:sum_index-1].mul(reporter_data[reporter_data.columns[sum_index-1]], axis=0)
	print_end_time(start_time)

	print "Reapplying crossover correction with new matrix",
	start_time = time.time()
	#apply the matrix multiplcation based on experiment
	reporter_data.iloc[0:,ms1_index+1:sum_index-1] = reporter_data.apply(dot_product_row, axis=1)
	print_end_time(start_time)

	print "Updating reporter sum",
	start_time = time.time()
	#update the reporter sum for each row
	reporter_data.iloc[0:,sum_index-1:sum_index] = reporter_data.iloc[0:,ms1_index+1:sum_index-1].sum(axis=1)
	print_end_time(start_time)

	#reapply the processed data back into the full dataframe
	ideaa_file.iloc[0:,0:sum_index] = reporter_data

	#get reporter sums for each MGF file
	# sums_by_mgf = pd.pivot_table(ideaa_file,index=["filename"], values=sum_headers, aggfunc="sum")
	
	#generate sums per experiment
	# for index, row in sums_by_mgf.iterrows():
	# 	exp = mgf_experiment_list[index]
	# 	experiment_sums[exp] = experiment_sums[exp].add(row)

	# Can add output if needed
	# print sums_by_mgf
	# print experiment_sums

	#renormalize the channels back to 1
	print "Normalizing channels",
	start_time = time.time()

	norm_df = reporter_data.iloc[0:,ms1_index+1:sum_index-1]
	norm_df = norm_df.div(norm_df.sum(axis=1), axis=0)
	reporter_data.iloc[0:,ms1_index+1:sum_index-1] = norm_df

	#reapply the processed data back into the full dataframe
	ideaa_file.iloc[0:,0:sum_index] = reporter_data

	#get all the totalnorm columns
	total_norm_columns = [col for col in ideaa_file.columns if "norm" in col]
	#find the index of the first and last norm so that we can grab them to overwrite the norm_totals
	total_norm_start_index = ideaa_file_indices.index(total_norm_columns[0])
	total_norm_end_index = ideaa_file_indices.index(total_norm_columns[-1])

	print_end_time(start_time)

	print "Normalizing totals",
	start_time = time.time()
	#apply the total normalization to the row and pull it out
	norm_totals_vals = reporter_data.apply(normalize_row_to_totals, axis=1).iloc[0:,sum_index:]
	print_end_time(start_time)
	
	#add these values back into the ideaa file and save it out
	ideaa_file.iloc[0:,total_norm_start_index:total_norm_end_index+1] = norm_totals_vals.values
	
	#write out each experiment to its own CSV file
	multiple_output_files = True

	print "Writing output files",
	start_time = time.time()
	if multiple_output_files:
		for row in ideaa_file.itertuples(index=False, name=None):
			experiment_name = mgf_experiment_list[row[0]]
			dot_index = ideaa_filename.index(".")
			experiment_output_filename = ideaa_filename[:dot_index]+"_batch_"+experiment_name+".txt"
			if not os.path.isfile(experiment_output_filename):
				output_file = open(experiment_output_filename, 'a')
				header_row = "\t".join(ideaa_file_indices)+"\n"
				output_file.write(header_row)
				output_files[experiment_output_filename] = output_file
			output_row = "\t".join([str(value) for value in row])
			output_files[experiment_output_filename].write(output_row+"\n")
	else:
			ideaa_file.to_csv(ideaa_filename+".out", sep='\t')
	
	for filename, output_file in output_files.iteritems():
		output_file.close()
	print_end_time(start_time)

	print "Complete"
	print "Total time: ",
	print_end_time(total_start)

def digest_experiments(experiment_file, ideaa_folder, sum_headers):
	with open(experiment_file, "r") as json_file:
   		experiments = json.load(json_file)
		for experiment in experiments:
			for mgf in experiment["mgf"]:
				mgf_crossover_list[mgf] = experiment["inverse_file"]
				mgf_experiment_list[mgf] = experiment["experiment"]
			summary_files[experiment["experiment"]] = experiment["summary_file"]
			inverse_filename = experiment["inverse_file"]
			crossover_data[inverse_filename] = read_in_crossover_file(inverse_filename, ideaa_folder)
			#initialize each experiment sum with an empty dataframe
			experiment_sums[experiment["experiment"]] = pd.DataFrame(0, index=range(1), columns = sum_headers)

def read_in_crossover_file(crossover_file, ideaa_folder):
	crossover_path = os.path.join(ideaa_folder, "SCRIPTS_FOR_GUI", "inverse_files", crossover_file)
	crossover = pd.read_csv(crossover_path, sep="\t")
	crossover = crossover.drop(crossover.columns[0], axis=1)
	return crossover

def dot_product_row(row):
	reporter_vals = pd.DataFrame([row[5:-1]])
	#print reporter_vals
	#print row[0]
	matrix = crossover_data[mgf_crossover_list[row[0]]]
	result = np.dot(reporter_vals, matrix)
	row.iloc[5:-1] = result[0]
	#floor of zero
	row.iloc[5:-1][row.iloc[5:-1] < 0] = 0
	return row

def normalize_row_to_totals(row):
	reporter_vals = pd.DataFrame([row[5:-1]])
	intensities = norm_total_intensities[mgf_experiment_list[row[0]]]
	#divide by the total intensites from all MGF files (extracted from summary)
	result = reporter_vals.div(intensities, axis=1)
	#normalize to row sum
	result = result.div(result.sum(axis=1), axis=0)
	row = row.append(result.iloc[0])
	return row

def read_intensities_from_summary_and_normalize():
	for exp, summary_filename in summary_files.iteritems():
		try:
			summary = open(summary_filename, "r")
			found_intensities = False
			while True:
				raw_line = summary.readline()
				#if we're at the end of the summary file, break the loop
				if raw_line == "" and found_intensities == True:
					break

				if raw_line == "Total Reporter Ion and MS1 Intensities per MGF file\n":
					found_intensities = True
					summary.readline()
					summary.readline()
					raw_line = summary.readline()

				if found_intensities == True:
					split_line = raw_line.split("\t")
					mgf_filename = split_line[0]
					intensity_str = split_line[2:]
					intensities = [float(intensity) for intensity in intensity_str]

					#if the mgf doesn't match an experiment, read the next line
					if not mgf_experiment_list.get(mgf_filename):
						continue
					#if the mgf matches one that belongs to the current experiment, add its intensity to the total
					if exp == mgf_experiment_list[mgf_filename]:
						if not experiment_intensities.get(exp):
							experiment_intensities[exp] = intensities
						#add the current intensity to the total intensity for this experiment
						else:
							experiment_intensities[exp] = [val1+val2 for val1, val2 in zip(experiment_intensities[exp], intensities)]

		except Exception as err:
 			print "Error reading from summary file"
 			print err
 			return

	for exp, intensity_list in experiment_intensities.iteritems():
		sum_int = sum(intensity_list)
		if sum_int != 0:
			norm_total_intensities[exp] = [intensity/sum_int for intensity in intensities]
		else:
			norm_total_intensities[exp] = intensities

def print_end_time(start_time):
	total_time = time.time()-start_time
	if total_time > 60:
		print " - " + str(total_time/60) + " mins"
	else:
		print " - " + str(total_time) + " secs"

if __name__ == "__main__":
	main()

