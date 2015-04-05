#ifndef READARRAY_H_
#define READARRAY_H_

#include <iostream>
#include <fstream>
#include <string>
#include <omp.h>
#include <stdio.h>
#include <eigen3\Eigen\Dense>
#include <vector>

Eigen::ArrayXXf readArray(std::string filename)
{
    int rows  = 0;
	std::vector<float> buff(0);
    std::ifstream infile;
    infile.open(filename);
    
	while (! infile.eof())
    {
        std::string line;
        getline(infile, line);
        std::stringstream stream(line);
		rows++;
        float x;
		while(stream >> x)
		{
			buff.push_back(x);
		}
    }
    infile.close();
	int cols = buff.size()/rows;
    Eigen::ArrayXXf result = (Eigen::Map<Eigen::ArrayXXf>(buff.data(),cols,rows)).transpose();
	
    return result;
}


#endif