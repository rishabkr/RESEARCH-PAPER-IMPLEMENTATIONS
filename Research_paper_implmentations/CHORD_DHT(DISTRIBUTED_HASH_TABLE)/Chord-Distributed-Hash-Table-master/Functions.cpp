#include "Headers.h"
ll getHashId(string key)
{
	 unsigned char ukey[key.length()+1];
	 //cout<<"Key:"<<key<<endl;
	 for(int i=0;i<key.length();i++)
	 	ukey[i]=key[i];
	 ukey[key.length()]='\0';
	 //<<"Ukey:"<<ukey<<endl;
	 unsigned char uhash[41];
	 char hash_str[41];
	 SHA1(ukey,sizeof(ukey),uhash);
	 //cout<<"uhash:"<<uhash<<endl;
	 for(int i=0;i<max_bit/8;i++)
	 {
	 	sprintf(hash_str+i,"%d",uhash[i]);
	 }
	 //cout<<"Hash string:"<<hash_str<<endl;
	 ll hashval=stoll(hash_str)%((ll)pow(2,max_bit));
	 return hashval;
}
// int establish_connection(nodeInfo node) //returns socket descriptor for communication
// {
//     string ip=node.ip;
//     int port=node.port;
//     struct sockaddr_in server_addr;
//     socklen_t len = sizeof(server_addr);
//     server_addr.sin_family = AF_INET;
//     server_addr.sin_addr.s_addr = inet_addr(ip.c_str());
//     server_addr.sin_port = htons(port);
//     int socket_num=socket(AF_INET,SOCK_DGRAM,0);
//     if(socket_num<0)
//     {
//         cerr<<"Error creating socket! \n";
//         exit(-1);
//     }
//     return socket_num;
// }	
int establish_connection(struct sockaddr_in &server_addr,nodeInfo node)
{
    string ip=node.ip;
    int port=node.port;
    socklen_t len = sizeof(server_addr);
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(ip.c_str());
    server_addr.sin_port = htons(port);
    int socket_num=socket(AF_INET,SOCK_DGRAM,0);
    if(socket_num<0)
    {
    	cerr<<node.ip<<"\t"<<node.port<<endl;
        cerr<<"Error creating socket! \n";
        exit(-1);
    }
    return socket_num;
}
vector<string> split_ip(string ipport) //returns vector string having ip at first postion and port at second
{								//both ip and port need to be typecast
	 vector<string>  ip_port;
    int pos = ipport.find(':');
    string ip = ipport.substr(0,pos);
    string port = ipport.substr(pos+1);
	ip_port.push_back(ip);
	ip_port.push_back(port);
    return ip_port;
}
