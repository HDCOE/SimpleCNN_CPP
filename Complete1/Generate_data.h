vector<tensor_t<float>> generate_dataset()
{
	vector<tensor_t<float>> input_all;
	tensor_t<float> input(2,2,1);
	tensor_t<float> input2(2,2,1);
	tensor_t<float> input3(2,2,1);
	tensor_t<float> input4(2,2,1);
	tensor_t<float> input5(2,2,1);
	tensor_t<float> input6(2,2,1);
	tensor_t<float> input7(2,2,1);
	tensor_t<float> input8(2,2,1);

// generate ab + cd 
//input 00 + 00 = 0
	input(0,0,0) = 0;
	input(0,1,0) = 0;
	input(1,0,0) = 0;
	input(1,1,0) = 0;
	input_all.push_back(input);

	input2(0,0,0) = 0;
	input2(0,1,0) = 0;
	input2(1,0,0) = 0;
	input2(1,1,0) = 1;
	input_all.push_back(input2);

	input3(0,0,0) = 0;
	input3(0,1,0) = 0;
	input3(1,0,0) = 1;
	input3(1,1,0) = 0;
	input_all.push_back(input3);

	input4(0,0,0) = 0;
	input4(0,1,0) = 0;
	input4(1,0,0) = 1;
	input4(1,1,0) = 1;
	input_all.push_back(input4);

	input5(0,0,0) = 0;
	input5(0,1,0) = 1;
	input5(1,0,0) = 0;
	input5(1,1,0) = 0;
	input_all.push_back(input5);

	input6(0,0,0) = 0;
	input6(0,1,0) = 1;
	input6(1,0,0) = 0;
	input6(1,1,0) = 1;
	input_all.push_back(input6);

	input7(0,0,0) = 0;
	input7(0,1,0) = 1;
	input7(1,0,0) = 1;
	input7(1,1,0) = 0;
	input_all.push_back(input7);

	input8(0,0,0) = 0;
	input8(0,1,0) = 1;
	input8(1,0,0) = 1;
	input8(1,1,0) = 1;
	input_all.push_back(input8);
/*
	input(0,0,0) = 1;
	input(0,1,0) = 0;
	input(1,0,0) = 0;
	input(1,1,0) = 0;
	input_all.push_back(input);

	input(0,0,0) = 1;
	input(0,1,0) = 0;
	input(1,0,0) = 0;
	input(1,1,0) = 1;
	input_all.push_back(input);

	input(0,0,0) = 1;
	input(0,1,0) = 0;
	input(1,0,0) = 1;
	input(1,1,0) = 0;
	input_all.push_back(input);

	input(0,0,0) = 1;
	input(0,1,0) = 0;
	input(1,0,0) = 1;
	input(1,1,0) = 1;
	input_all.push_back(input);

	input(0,0,0) = 1;
	input(0,1,0) = 1;
	input(1,0,0) = 0;
	input(1,1,0) = 0;
	input_all.push_back(input);

	input(0,0,0) = 1;
	input(0,1,0) = 1;
	input(1,0,0) = 0;
	input(1,1,0) = 1;
	input_all.push_back(input);

	input(0,0,0) = 1;
	input(0,1,0) = 1;
	input(1,0,0) = 1;
	input(1,1,0) = 0;
	input_all.push_back(input);

	input(0,0,0) = 1;
	input(0,1,0) = 1;
	input(1,0,0) = 1;
	input(1,1,0) = 1;
	input_all.push_back(input);
*/
//print_tensor(input);

return input_all;
}

vector<tensor_t<float>>  generate_output()
{
	
	vector<tensor_t<float>> Y;
	tensor_t<float> y_out1(1,1,2);
	tensor_t<float> y_out2(1,1,2);
	tensor_t<float> y_out3(1,1,2);
	tensor_t<float> y_out4(1,1,2);
	tensor_t<float> y_out5(1,1,2);
	tensor_t<float> y_out6(1,1,2);
	tensor_t<float> y_out7(1,1,2);
	tensor_t<float> y_out8(1,1,2);

// input ab+cd :00+00
// output z is 1 when y0 =1, z is 0 when y1 =1
	y_out1(0,0,0) = 0;
	y_out1(0,0,1) = 1;
	Y.push_back(y_out1);

	y_out2(0,0,0) = 0;
	y_out2(0,0,1) = 1;
	Y.push_back(y_out2);

	y_out3(0,0,0) = 0;
	y_out3(0,0,1) = 1;
	Y.push_back(y_out3);

	y_out4(0,0,0) = 1;
	y_out4(0,0,1) = 0;
	Y.push_back(y_out4);

	y_out5(0,0,0) = 1;
	y_out5(0,0,1) = 0;
	Y.push_back(y_out5);

	y_out6(0,0,0) = 1;
	y_out6(0,0,1) = 0;
	Y.push_back(y_out6);

	y_out7(0,0,0) = 1;
	y_out7(0,0,1) = 0;
	Y.push_back(y_out7);

	y_out8(0,0,0) = 1;
	y_out8(0,0,1) = 0;
	Y.push_back(y_out8);
/*
	y_out(0,0,0) = 1;
	y_out(0,0,1) = 0;
	Y.push_back(y_out);

	y_out(0,0,0) = 1;
	y_out(0,0,1) = 0;
	Y.push_back(y_out);

	y_out(0,0,0) = 1;
	y_out(0,0,1) = 0;
	Y.push_back(y_out);

	y_out(0,0,0) = 1;
	y_out(0,0,1) = 0;
	Y.push_back(y_out);

	y_out(0,0,0) = 1;
	y_out(0,0,1) = 0;
	Y.push_back(y_out);

	y_out(0,0,0) = 1;
	y_out(0,0,1) = 0;
	Y.push_back(y_out);

	y_out(0,0,0) = 1;
	y_out(0,0,1) = 0;
	Y.push_back(y_out);

	y_out(0,0,0) = 1;
	y_out(0,0,1) = 0;
	Y.push_back(y_out);
*/

	return Y;

}
struct dataset
{
	tensor_t<float> data;
	tensor_t<float> y_out;
};

vector<dataset> Dataset_and()
{
	
	dataset set {tensor_t<float>( 2, 2, 1 ), tensor_t<float>( 1, 1, 2 )};
	vector<dataset> cases;
	// ab
	set.data(0,0,0) = 0;
	set.data(0,1,0) = 0;
    // cd
	set.data(1,0,0) = 0;
	set.data(1,1,0) = 0;

	set.y_out(0,0,0) = 0; //y1 = ab
	set.y_out(0,0,1) = 1; //y2 = ab
	cases.push_back(set);

    dataset set1 {tensor_t<float>( 2, 2, 1 ), tensor_t<float>( 1, 1, 2 )};
		// ab
	set1.data(0,0,0) = 0;
	set1.data(0,1,0) = 1;
    // cd
	set1.data(1,0,0) = 0;
	set1.data(1,1,0) = 1;

	set1.y_out(0,0,0) = 0; //y1 = ab
	set1.y_out(0,0,1) = 1; //y2 = ab
cases.push_back(set1);

dataset set2 {tensor_t<float>( 2, 2, 1 ), tensor_t<float>( 1, 1, 2 )};
		// ab
	set2.data(0,0,0) = 1;
	set2.data(0,1,0) = 0;
    // cd
	set2.data(1,0,0) = 1;
	set2.data(1,1,0) = 0;

	set2.y_out(0,0,0) = 1; //y1 = ab
	set2.y_out(0,0,1) = 0; //y2 = ab
	cases.push_back(set2);

dataset set3 {tensor_t<float>( 2, 2, 1 ), tensor_t<float>( 1, 1, 2 )};
		// ab
	set3.data(0,0,0) = 1;
	set3.data(0,1,0) = 1;
    // cd
	set3.data(1,0,0) = 1;
	set3.data(1,1,0) = 1;

	set3.y_out(0,0,0) = 0; //y1 = ab
	set3.y_out(0,0,1) = 1; //y2 = ab
	cases.push_back(set3);
return cases;

}
void Dataset_and_2(vector<dataset>& cases)
{
	
	dataset set {tensor_t<float>( 2, 2, 1 ), tensor_t<float>( 1, 1, 2 )};
//	vector<dataset> cases;
	// ab
	set.data(0,0,0) = 0;
	set.data(0,1,0) = 0;
    // cd
	set.data(1,0,0) = 0;
	set.data(1,1,0) = 0;

	set.y_out(0,0,0) = 0; //y1 = ab
	set.y_out(0,0,1) = 1; //y2 = ab
	cases.push_back(set);

dataset set1 {tensor_t<float>( 2, 2, 1 ), tensor_t<float>( 1, 1, 2 )};
		// ab
	set1.data(0,0,0) = 0;
	set1.data(0,1,0) = 1;
    // cd
	set1.data(1,0,0) = 0;
	set1.data(1,1,0) = 1;

	set1.y_out(0,0,0) = 0; //y1 = ab
	set1.y_out(0,0,1) = 1; //y2 = ab
	cases.push_back(set1);

dataset set2 {tensor_t<float>( 2, 2, 1 ), tensor_t<float>( 1, 1, 2 )};
		// ab
	set2.data(0,0,0) = 1;
	set2.data(0,1,0) = 0;
    // cd
	set2.data(1,0,0) = 1;
	set2.data(1,1,0) = 0;

	set2.y_out(0,0,0) = 1; //y1 = ab
	set2.y_out(0,0,1) = 0; //y2 = ab
	cases.push_back(set2);

dataset set3 {tensor_t<float>( 2, 2, 1 ), tensor_t<float>( 1, 1, 2 )};
		// ab
	set3.data(0,0,0) = 1;
	set3.data(0,1,0) = 1;
    // cd
	set3.data(1,0,0) = 1;
	set3.data(1,1,0) = 1;

	set3.y_out(0,0,0) = 0; //y1 = ab
	set3.y_out(0,0,1) = 1; //y2 = ab
	cases.push_back(set3);
}

vector<dataset> Dataset_AND2()
{
	vector<dataset> cases;
	dataset set {tensor_t<float>( 1, 1, 2 ), tensor_t<float>( 1, 1, 1 )};
	// ab
	set.data(0,0,0) = 0;
	set.data(0,0,1) = 0;

	set.y_out(0,0,0) = 0; //y1 = ab
	cases.push_back(set);

    dataset set1 {tensor_t<float>( 1, 1, 2 ), tensor_t<float>( 1, 1, 1 )};
		// ab
	set1.data(0,0,0) = 0;
	set1.data(0,0,1) = 1;

	set1.y_out(0,0,0) = 0; //y1 = ab
	cases.push_back(set1);

dataset set2 {tensor_t<float>( 1, 1, 2 ), tensor_t<float>( 1, 1, 1 )};
		// ab
	set2.data(0,0,0) = 1;
	set2.data(0,0,1) = 0;

	set2.y_out(0,0,0) = 0; //y1 = ab
	cases.push_back(set2);

dataset set3 {tensor_t<float>( 1, 1, 2 ), tensor_t<float>( 1, 1, 1 )};
		// ab
	set3.data(0,0,0) = 1;
	set3.data(0,0,1) = 1;
    // cd
	set3.y_out(0,0,0) = 1; //y1 = ab
	cases.push_back(set3);
return cases;

}