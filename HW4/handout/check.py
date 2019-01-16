def check(output_file, target_file):
	print(output_file)
	output = open(output_file, "r")
	target = open(target_file, "r")
	output_lines = output.readlines()
	target_lines = target.readlines()
	for o, t in zip(output_lines, target_lines):
		assert o == t
	print("Pass checking...")

if __name__ == "__main__":
	check("my_result/model2_formatted_train.tsv", "largeoutput/model2_formatted_train.tsv")
	check("my_result/model2_formatted_valid.tsv", "largeoutput/model2_formatted_valid.tsv")
	check("my_result/model2_formatted_test.tsv", "largeoutput/model2_formatted_test.tsv")