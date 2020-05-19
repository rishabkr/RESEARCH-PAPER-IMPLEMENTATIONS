#include<iostream>
#include<math.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <ctime>
#include<unistd.h>
#include<cstring>
#include<bits/stdc++.h>
#include <openssl/sha.h>
#include <thread>
#define ll long long int
#define BUF_MAX 4096
#define max_bit 16
#define lst_sz 8
using namespace std;
struct nodeInfo
{
	string ip;
	int port;
	ll id;
};
