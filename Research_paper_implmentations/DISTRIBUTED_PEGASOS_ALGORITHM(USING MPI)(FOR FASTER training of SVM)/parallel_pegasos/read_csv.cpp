#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm> 
#include <cctype>
#include <locale>

using namespace std;
static int csv_rows=-1, csv_cols=-1;

static inline void ltrim(string &s) {
    s.erase(s.begin(), find_if(s.begin(), s.end(), [](int ch) {
        return !isspace(ch);
    }));
}

static inline void rtrim(string &s) {
    s.erase(find_if(s.rbegin(), s.rend(), [](int ch) {
        return !isspace(ch);
    }).base(), s.end());
}

static inline void trim(std::string &s) {
    ltrim(s);
    rtrim(s);
}

void read_csv(string fname, vector<vector<string>>& data)
{
	fstream f ;
	f.open(fname, ios::in);

	vector<string> row ;
	string line ;
	while(getline(f, line, '\n'))
	{
		stringstream ss(line) ;
		string word; 
		vector<string> row ;
		while(getline(ss, word, ','))
		{
			trim(word);
			row.push_back(word) ;
		}
		data.push_back(row) ;
	}
	f.close();
}

void transform_data(vector<vector<string>>& data, vector<vector<double>>& out, bool delete_original=true)
{
	for(const vector<string>& row : data)
	{
		vector<double> rr; 
		for(const string& value : row)
		{
			rr.push_back(stod(value));
		}
		out.emplace_back(rr);
	}
	if(delete_original)
		data.clear();
}

double** alloc2d(int rows, int cols)
{
	double *t = new double[rows*cols] ;	
	double** A = new double*[rows] ;
	for(int i=0; i<rows; ++i)
	{
		A[i]=t+(i*cols);	
	}
	return A;
}
void dealloc2d(double **arr, int rows)
{
	delete[] arr[0];
	delete[] arr ;
}

double** transform_data_primitive(vector<vector<string>>& data, bool delete_original=true)
{
	int rows = data.size() ;
	if(rows==0) return nullptr;
	int cols = data[0].size() ;
	if(cols==0) return nullptr; 

	//update the global csv_rows and csv_cols to the actual size
	csv_rows=rows;
	csv_cols=cols;
	
	double** arr = alloc2d(rows, cols);
	for(int i=0; i<rows; ++i)
	{
		for(int j=0; j<cols; ++j)
		{
			arr[i][j]=stod(data[i][j]);
		}
	}
	if(delete_original)
		data.clear();
	return arr;
}

//int main()
//{
//	string fname ;	
//	cout << "Enter the file name : ";
//	cin >> fname;
//
//	vector<vector<string>> data ;
//	read_csv(fname, data);
//	for(const vector<string>& row : data)
//	{
//		for(const string& val : row)
//			cout << val << '~' ;
//		cout << '\n' ;
//	}
//
//	double** ddata ;
//	cout << "init size : " << data.size() << '\n' ;
//	ddata = transform_data_primitive(data);
//	cout << "final size : " << data.size() << '\n' ;
//	cout << csv_rows << ' ' << csv_cols << '\n';
//
//	for(int i=0; i<csv_rows; ++i)
//	{
//		for(int j=0; j<csv_cols; ++j)
//		{
//			cout << ddata[i][j] << '_' ;
//		}
//		cout << '\n';
//	}
//	dealloc2d(ddata, csv_rows);
//}
