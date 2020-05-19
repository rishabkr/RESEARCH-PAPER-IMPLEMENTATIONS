#include "read_csv.cpp"
#include <random>
#include <cstdlib>
#include <cstring>
#include <cmath>

double** x_train;
double** y_train;
double** x_test;
double** y_test;
double** K;

int train_rows, train_cols;
int test_rows, test_cols;

class Weight
{
public:
	double *w ;
	int sz ;
	Weight()
	{
		sz=-1;
	}
	void generate(int n, int v=0)
	{
		if(sz!=-1)
		{
			delete[] w;
		}
		w=new double[n] ;
		sz=n;
		for(int i=0; i<n; ++i) w[i]=v;
	}
	Weight(int n, int v=0)
	{
		sz=-1;
		generate(n, v);
	}	
	~Weight()
	{
		delete[] w;
	}
};

double dot_vector(double *a, double *b, int l)
{
	double res=0;
	for(int i=0; i<l; ++i) res+=(a[i]*b[i]);	
	return res;
}

double kernel(double* x1, double *x2, int l, double gamma)
{
	double acc=0 ;
	for(int i=0; i<l; ++i)
	{
		acc+=(x1[i]-x2[i])*(x1[i]-x2[i]) ;
	}
	return exp(-gamma*acc) ;
}

void vector_scalar_mult(double scalar, double *v, int l)
{
	for(int i=0; i<l; ++i) v[i]*=scalar;
}

void add_vectors(double *v1, double *v2, double *res, int l)
{
	for(int i=0; i<l; ++i)
	{
		res[i]=v1[i]+v2[i];
	}
}

void load_data()
{
	vector<vector<string>> buffer;
	
	read_csv("bc_train_data.csv", buffer) ;
	x_train=transform_data_primitive(buffer);
	train_rows=csv_rows;
	train_cols=csv_cols;

	read_csv("bc_test_data.csv", buffer) ;
	x_test=transform_data_primitive(buffer);
	test_rows=csv_rows;
	test_cols=csv_cols;

	read_csv("bc_train_labels.csv", buffer);
	y_train=transform_data_primitive(buffer);

	read_csv("bc_test_labels.csv", buffer);
	y_test=transform_data_primitive(buffer);

	cout << "Data Loaded with following info :\n";
	cout << "Train Data Dimensions : " << train_rows << " x " << train_cols << '\n' ;
	cout << "Test Data Dimensions : " << test_rows << " x " << test_cols << '\n' ;
}

void convert_to_sign(double** a, int r, int c)
{
	for(int i=0; i<r; ++i)
		for(int j=0; j<c; ++j)
			if(a[i][j]==0) a[i][j]=-1;	
}

void train(int num_iterations, double regularization, Weight &weight)
{
	random_device rd; 
	mt19937 rng(rd());
	uniform_int_distribution<int> uni(0, train_rows-1); 

	K = alloc2d(train_rows, train_rows);
	for(int i=0; i<train_rows; ++i)
	{
		for(int j=0; j<train_rows; ++j)
		{
			K[i][j]=kernel(x_train[i], x_train[j], train_cols, 1);
		}
		if(i%1000==0)
			cout << "Done " << i/1000 << "\r" ;
	}

	weight.generate(train_rows);	
	double x[train_cols];
	for(int i=0; i<num_iterations; ++i)
	{
		int j=uni(rng);		
		memcpy(x, x_train[j], train_cols*sizeof(double));
		double y=y_train[j][0];
		//double check=y*dot_vector(x, weight.w, train_cols);
		double check=0;
		for(int k=0; k<train_rows; ++k)
		{
			check += weight.w[k] * y * K[j][k] ;
		}
		check *= y/regularization ;
		if(check<1)
		{
			weight.w[j]++ ;
		}
		if(i%1000==0)
		{
			cout << "Done : " << i/1000 << "\r" ;
		}
	}
	cout << "Training Complete.\n";
}

double predict(double *v, Weight& w)
{
	double prediction=0;
	for(int i=0; i<train_rows; ++i)
	{
		prediction += w.w[i] * y_train[i][0] * kernel(x_train[i], v, train_cols, 10);
	}
	if(prediction < 0)
		return -1;
	return 1;
}

int main()
{
	load_data();
	convert_to_sign(y_train, train_rows, 1);
	convert_to_sign(y_test, test_rows, 1);
	Weight weight;
	train(20*train_rows, 0.8, weight);
	cout << "Weights : " ;
	for(int i=0; i<weight.sz; ++i)
		cout << weight.w[i] << ' ';
	cout << '\n';

	double acc=0;
	for(int i=0; i<test_rows; ++i)
	{
		double actual=y_test[i][0];
		double predicted=predict(x_test[i], weight);
		cout << actual << ' ' << predicted << '\n' ;
		acc += (actual==predicted);
	}
	cout << "Accuracy : " << acc/test_rows << '\n';
	dealloc2d(x_train, train_rows);
	dealloc2d(x_test, test_rows);
	dealloc2d(y_train, train_rows);
	dealloc2d(y_test, test_rows);
}
