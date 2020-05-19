#include "read_csv.cpp"
#include <mpi.h>
#include <random>
#include <cstdlib>
#include <cstring>

double** x_train;
double** y_train;
double** x_test;
double** y_test;

int train_rows, train_cols;
int test_rows, test_cols;
int n_cpu, pid;

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
	
	read_csv("train_data_l.csv", buffer) ;
	x_train=transform_data_primitive(buffer);
	train_rows=csv_rows;
	train_cols=csv_cols;

	read_csv("test_data_l.csv", buffer) ;
	x_test=transform_data_primitive(buffer);
	test_rows=csv_rows;
	test_cols=csv_cols;

	read_csv("train_labels_l.csv", buffer);
	y_train=transform_data_primitive(buffer);

	read_csv("test_labels_l.csv", buffer);
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
	MPI_Status status;

	if(pid!=0)
	{
		x_train=alloc2d(train_rows, train_cols);
		y_train=alloc2d(train_rows, 1);
		x_test=alloc2d(test_rows, test_cols);
		y_test=alloc2d(test_rows, 1);
	}

	MPI_Bcast(&x_train[0][0], train_rows*train_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&y_train[0][0], train_rows*1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&x_test[0][0], test_rows*test_cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&y_test[0][0], test_rows*1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	weight.generate(train_cols);	
	int iter_for_slave=0, iter_for_root;
	if(n_cpu>1)
		iter_for_slave = num_iterations/n_cpu;
	iter_for_root=num_iterations - iter_for_slave*(n_cpu-1);
	int limit = (pid == 0) ? iter_for_root : iter_for_slave ;
	printf("Pid : %d   limit : %d\n", pid, limit);
	for(int i=0; i<limit; ++i)
	{
		double eta=1/(regularization*(i+1));
		int j=uni(rng);		
		double x[train_cols];
		memcpy(x, x_train[j], train_cols*sizeof(double));
		double y=y_train[j][0];
		double check=y*dot_vector(x, weight.w, train_cols);
		double multiplier=1-eta*regularization;
		vector_scalar_mult(multiplier, weight.w, train_cols) ;	
		if(check<1)
		{
			vector_scalar_mult(eta*y, x, train_cols);
			add_vectors(weight.w, x, weight.w, train_cols);	
		}
	}
	//should put a barrier
	//MPI_Barrier(MPI_COMM_WORLD);
	if(pid==0)
	{
		//if we are root.
		//we have our own weight as weight.
		for(int i=1; i<n_cpu; ++i)
		{
			double temp[train_cols] ;	
			MPI_Recv(&temp[0], train_cols, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &status);
			printf("Pid : %d recived data \n", i);
			for(int i=0; i<train_cols; ++i)
			{
				weight.w[i]+=temp[i] ;
			}
		}
		for(int i=0; i<train_cols; ++i)
		{
			weight.w[i]/=n_cpu ;
		}
		printf("Training complete at root\n");
	} else
	{
		//if we are slave
		MPI_Send(&weight.w[0], train_cols, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		printf("Pid : %d sent data to root\n", pid);
	}
}

double predict(double *v, Weight& w)
{
	double x=dot_vector(v, w.w, w.sz);
	if(x<0) return -1;
	else return 1;
}

int main(int argc, char* argv[])
{

    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &pid);
    MPI_Comm_size(MPI_COMM_WORLD, &n_cpu);

	Weight weight;
	if(pid==0)
	{
		load_data();
	}
	MPI_Bcast(&train_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&train_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&test_rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(&test_cols, 1, MPI_INT, 0, MPI_COMM_WORLD);
	
	train(50*train_rows, 0.8, weight);
//	printf("pid : %d %d\n", pid, train_rows);
//	printf("pid : %d %d\n", pid, train_cols);
//	printf("pid : %d %d\n", pid, test_rows);
//	printf("pid : %d %d\n", pid, test_cols);

	if(pid==0)
	{
		cout << "Weights : " ;
		for(int i=0; i<weight.sz; ++i)
			cout << weight.w[i] << ' ';
		cout << '\n';

		double acc=0;
		for(int i=0; i<test_rows; ++i)
		{
			double actual=y_test[i][0];
			double predicted=predict(x_test[i], weight);
//			cout << actual << ' ' << predicted << '\n' ;
			acc += (actual==predicted);
		}
		cout << "Accuracy : " << acc/test_rows << '\n';
	}
	dealloc2d(x_train, train_rows);
	dealloc2d(x_test, test_rows);
	dealloc2d(y_train, train_rows);
	dealloc2d(y_test, test_rows);
	MPI_Finalize();
}
