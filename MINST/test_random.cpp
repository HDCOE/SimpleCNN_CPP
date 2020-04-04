#include <random>
#include <stdio.h>
#include <stdlib.h> 
#include<iostream>
#include <fstream>
using namespace std;

double unifRand()
{
    return rand() / double(RAND_MAX);
}
//
double unifRand(double a, double b)
{
    return (b-a)*unifRand() + a;
}

uint8_t* write_file(const char* file_name)
{
	ofstream file(file_name);

	if (file.is_open())
	{
		file << "test \n";
		file << "test2 \n";
		for (int i = 0; i < 20; ++i)
		{
			printf("%.2f\n", unifRand(-2,2) );
			file << unifRand(-2,2) << std::endl;
		}
	}
	file.close();
}

void print_t(const char * input)
{
  	std::cout << input << std::endl;
}

int main(int argc, char const *argv[])
{
	for (int i = 0; i < 20; ++i)
	{
		printf("%.2f\n", unifRand(-2,2) );
	}

	print_t("test");
    write_file("test.txt");
	return 0;
}