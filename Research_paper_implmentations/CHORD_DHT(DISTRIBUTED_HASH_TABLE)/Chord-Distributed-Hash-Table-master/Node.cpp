#include "Functions.cpp"
class Node
{
	public:
	int sockid;
	vector<nodeInfo> successorList;
	vector<nodeInfo> fingerTable;
	bool inRing;
	map<ll,string> keyStore;
	nodeInfo currentNode;
	nodeInfo predecessor,successor;
	void initializeNode()
	{
		struct sockaddr_in address;
		srand(time(NULL));
		int port=rand()%65536;//between 0 to 65535
		if(port<1024)
			port+=1024;	//0-1023 are system ports
			
		socklen_t len=sizeof(address);
		address.sin_family=AF_INET;
		address.sin_port=htons(port);
		address.sin_addr.s_addr=inet_addr("127.0.0.1");
		sockid=socket(AF_INET,SOCK_DGRAM,0);
		if(bind(sockid,(struct sockaddr *)&address,len)<0)
		{
			perror("Bind failed");
			exit(-1);
		}
		cout<<"Port number "<<port<<" assigned"<<endl;
		currentNode.ip=inet_ntoa(address.sin_addr);
		currentNode.port=port;
		fingerTable=vector<nodeInfo>(max_bit);
		successorList=vector<nodeInfo>(lst_sz);
		inRing=false;
	}
	void createRing()
	{
		if(inRing)
		{
			cout<<"Node is already in a ring"<<endl;
			return;
		}
		string ip=currentNode.ip;
		string key=ip+":"+to_string(currentNode.port);
		currentNode.id=getHashId(key);  // ID assigned
		cout<<"Id assigned "<<currentNode.id<<endl;

		//Setting Successor info as self
		successor.ip=ip;
		successor.port=currentNode.port;
		successor.id=currentNode.id;

		//Successor List??
		for(int i=0;i<lst_sz;i++)
		{
			successorList[i].ip=ip;
			successorList[i].port=currentNode.port;
			successorList[i].id=currentNode.id;
		}

		//Setting predecessor as nill
		predecessor.ip=currentNode.ip;
		predecessor.port=currentNode.port;
		predecessor.id=currentNode.id;
		//https://stackoverflow.com/questions/14575068/close-function-not-bei               


		//Setting finger table, all reference to own
		for(int i=0;i<max_bit;i++)
		{
			fingerTable[i].ip=ip;
			fingerTable[i].port=currentNode.port;
			fingerTable[i].id=currentNode.id;
		}
		//Start listening to other nodes
		thread listener(&Node::startListening,this);
		listener.detach();
		//stabilize, updates finger table periodically
		thread stabiliser(&Node::stabilise,this);
		stabiliser.detach();
		//Node is now in the ring
		inRing=true;
	}
	

	void startListening()
	{
		struct sockaddr_in client;
		socklen_t len=sizeof(client);
		while(1)
		{
			char message[BUF_MAX+1];
			int msgsize=recvfrom(sockid,message,BUF_MAX,0,(struct sockaddr *)&client,&len);
			message[msgsize]='\0';
			string received=message;
			thread exec_thread(&Node::exec_comm,this,client,received);
			exec_thread.detach();
		}
	}	
	void exec_comm(struct sockaddr_in client,string message) 
	{ 
		//cout<<"Received:"<<message<<endl;
       if(message.substr(0,3)=="put")
		{
			string keyval=message.substr(4);
			ll key=stoll(keyval.substr(0,keyval.find(':')));
			string value=keyval.substr(keyval.find(':')+1);
			keyStore[key]=value;
			cout<<"this is value put"<<endl;
			cout<<keyStore[key]<<endl;
		}
		else if(message.substr(0,5)=="finds")
		{
			string ids=message.substr(6);
			nodeInfo suc=find_successor(stoll(ids));
			socklen_t client_size=sizeof(client);
			char ipPort[40];
			strcpy(ipPort,(suc.ip+":"+to_string(suc.port)).c_str());
			//cout<<"Sending: "<<ipPort<<endl;
			sendto(sockid,ipPort,strlen(ipPort),0,(struct sockaddr*) &client,client_size);
		}
		else if(message.substr(0,5)=="findp")
		{
			socklen_t client_size=sizeof(client);
			char ipPort[40];
			strcpy(ipPort,(predecessor.ip+":"+to_string(predecessor.port)).c_str());
			//cout<<"Sending: "<<ipPort<<endl;
			sendto(sockid,ipPort,strlen(ipPort),0,(struct sockaddr*) &client,client_size);
		}
		else if(message.substr(0,7)=="changes")
		{
			string ids=message.substr(8);
			vector<string> splitip=split_ip(ids);
			successor.ip=splitip[0];
			successor.port=atoi(splitip[1].c_str());
			successor.id=getHashId(ids);
			// fingerTable[0].ip=successor.ip;
			// fingerTable[0].port=successor.port;
			// fingerTable[0].id=successor.id;
		}
		else if(message.substr(0,7)=="changep")
		{
			string ids=message.substr(8);
			vector<string> splitip=split_ip(ids);
			predecessor.ip=splitip[0];
			predecessor.port=atoi(splitip[1].c_str());
			predecessor.id=getHashId(ids);
		}
    } 
	void display()
	{
		cout<<"Current Node"<<endl;
		cout<<"IP "<<currentNode.ip<<endl;
		cout<<"Port "<<currentNode.port<<endl;
		cout<<"Id "<<currentNode.id<<endl;
		cout<<"Successor Node"<<endl;
		cout<<"IP "<<successor.ip<<endl;
		cout<<"Port "<<successor.port<<endl;
		cout<<"Id "<<successor.id<<endl;
		cout<<"Predecessor Node"<<endl;
		cout<<"IP "<<predecessor.ip<<endl;
		cout<<"Port "<<predecessor.port<<endl;
		cout<<"Id "<<predecessor.id<<endl;
		cout<<"Finger table:"<<endl;
		for(int i=0;i<max_bit;i++)
		{
			cout<<fingerTable[i].ip<<"\t"<<fingerTable[i].port<<"\t"<<fingerTable[i].id<<endl;
		}
		cout<<"Key value pairs::"<<endl;
		for (map<ll,string>:: iterator it = keyStore.begin(); it != keyStore.end(); it++)
        	cout<<it->first<<"\t"<<it->second<<endl; 
	}


	void joinRing(string new_ip,int new_port)
	{
		struct sockaddr_in server;
	    socklen_t len = sizeof(server);
	    //cout<<new_ip<<endl;
	    //cout<<new_port<<endl;
	    nodeInfo new_node;
	   	string key=currentNode.ip+":"+to_string(currentNode.port);
		currentNode.id=getHashId(key);  // ID assigned
		cout<<"Id assigned: "<<currentNode.id<<endl;
	    new_node.ip=new_ip;
	    
	    new_node.port=new_port;
	    
	    int new_sock = establish_connection(server,new_node);

	    string current_ip=currentNode.ip;
	    
	    int current_port=currentNode.port;

	    ll node_id=currentNode.id;

	    char find_query[46];

	    strcpy(find_query,("finds:"+to_string(node_id)).c_str());
	    //cout<<"Sending "<<find_query<<endl;
	    if (sendto(new_sock, find_query, strlen(find_query), 0, (struct sockaddr*) &server, len) == -1)
	    {
	        perror("error");
	        exit(-1);
	    }
	    //cout<<"sent"<<endl;
	    char ip_and_port[41];
	    int length;
	    if ((length = recvfrom(new_sock, ip_and_port, 1024, 0, (struct sockaddr *) &server, &len)) == -1)
	    {
	        perror("error");
	        exit(-1);
	    }
	   cout<<"Received "<<ip_and_port<<endl;
	    ip_and_port[length] = '\0';
	    close(new_sock);
	   	string ipPort=ip_and_port;
	    vector<string> splitip=split_ip(ipPort);
	    successor.ip=splitip[0];
	    successor.port=atoi(splitip[1].c_str());
	    successor.id=getHashId(ipPort);
	    //initialise finger table as all entries pointing to immediate successor
	   	for(int i=0;i<max_bit;i++)
		{
			fingerTable[i].ip=successor.ip;
			fingerTable[i].port=successor.port;
			fingerTable[i].id=successor.id;
		}
	   	//ask for predecessor of this node
	    nodeInfo node=ask_predecessor(successor);
	   	// ask this node to change its predecessor
	   	predecessor.ip=node.ip;
	   	predecessor.port=node.port;
	   	predecessor.id=node.id;
	    ask_change_predecessor(successor,currentNode);
	   	//ask this node's old predecessor to change its successor
	    ask_change_successor(node,currentNode);
	    inRing=true;
	    // stabilise, updates finger table periodically
	    thread stabiliser(&Node::stabilise,this);
		stabiliser.detach();
	    thread listener(&Node::startListening,this);
	    listener.detach();
 	}

 	void stabilise()
 	{
 		while(1)
 		{
 			std::this_thread::sleep_for(std::chrono::seconds(1));
 			setFingerTable();
 		}
 	}
 	nodeInfo ask_predecessor(nodeInfo node)
 	{
 		struct sockaddr_in server;
	    socklen_t len = sizeof(server);
	    int new_sock = establish_connection(server,node);
	    char change_query[6];
	    strcpy(change_query,string("findp").c_str());
	    //cout<<"Sending: "<<change_query<<endl;
	    if (sendto(new_sock, change_query, strlen(change_query), 0, (struct sockaddr*) &server, len) == -1)
	    {
	        perror("error");
	        exit(-1);
	    }
	    char ip_and_port[41];
	    int length;
	    if ((length = recvfrom(new_sock, ip_and_port, 1024, 0, (struct sockaddr *) &server, &len)) == -1)
	    {
	        perror("error");
	        exit(-1);
	    }
	    ip_and_port[length] = '\0';
	    //cout<<"Received: "<<ip_and_port<<endl;
	    close(new_sock);
	    string ipPort=ip_and_port;
	    vector<string> splitip=split_ip(ipPort);
	    nodeInfo ret;
	    ret.ip=splitip[0];
	    ret.port=atoi(splitip[1].c_str());
	    ret.id=getHashId(ipPort);
	    return ret;
 	}


 	void ask_change_predecessor(nodeInfo destNode, nodeInfo targetNode)
 	{
 		struct sockaddr_in server;
	    socklen_t len = sizeof(server);
	    int new_sock = establish_connection(server,destNode);
	    char change_query[49];
	    strcpy(change_query,string("changep:"+targetNode.ip+":"+to_string(targetNode.port)).c_str());
	   // cout<<"Sending: "<<change_query<<endl;
	    if (sendto(new_sock, change_query, strlen(change_query), 0, (struct sockaddr*) &server, len) == -1)
	    {
	        perror("error");
	        exit(-1);
	    }
	    close(new_sock);
 	}


 	void ask_change_successor(nodeInfo destNode, nodeInfo targetNode)
 	{
 		struct sockaddr_in server;
	    socklen_t len = sizeof(server);
	    int new_sock = establish_connection(server,destNode);
	    char change_query[40];
	    //cout<<"Sending: "<<change_query<<endl;
	    strcpy(change_query,string("changes:"+targetNode.ip+":"+to_string(targetNode.port)).c_str());
	    if (sendto(new_sock, change_query, strlen(change_query), 0, (struct sockaddr*) &server, len) == -1)
	    {
	        perror("error");
	        exit(-1);
	    }
	    
	    close(new_sock);
 	}

    
 	nodeInfo find_successor(ll targetnode_id) //target node id is the joining node id
	{
//nodeInfo successor and predecessor are stored at each node
	if(targetnode_id > currentNode.id && targetnode_id <= successor.id || successor.id<currentNode.id && (targetnode_id<successor.id || targetnode_id>currentNode.id))//value lies in proper range
	{
		return successor;
	}

	else if(currentNode.id==successor.id || targetnode_id==currentNode.id)//only 1 element in the ring
	{
		return currentNode;
	}

	else if(successor.id==predecessor.id)//only 2 elements case
	{
		return currentNode;
	}
	else
	{
		nodeInfo preceding_node=closest_preceding_node(targetnode_id);//O(log n) lookup
		if(targetnode_id==currentNode.id)
		{
			return successor;
		}
		else
		{
			struct sockaddr_in server_addr;
		    socklen_t len = sizeof(server_addr);
		    if(preceding_node.ip=="") // couldn't find preceding node
		    	preceding_node=successor;
		    int sock=sock=establish_connection(server_addr,preceding_node);
		    char node_id[46];
		    strcpy(node_id,("finds:"+to_string(targetnode_id)).c_str());
		    sendto(sock,node_id, strlen(node_id), 0, (struct sockaddr*)&server_addr, len);

		    //receive the successor node from this node
		    char recvipport[41];

			int recvdbytes = recvfrom(sock, recvipport, 1024, 0, (struct sockaddr *) &server_addr, &len);
			close(sock);
			recvipport[recvdbytes]='\0';
			nodeInfo temp;
			if(recvdbytes<0)
			{
				temp.ip="";
				temp.port=-1;
				temp.id=-1;
				return temp;
			}

			string ipPort=recvipport;
		    ll id = getHashId(ipPort);
		    vector<string> splitip=split_ip(ipPort);
		    temp.ip = splitip[0];
		    temp.port = atoi(splitip[1].c_str());
		    temp.id = id;
		    return temp;
		}

	}
	}
	void setFingerTable()
	{

		int next = 0;
		long long int mod = pow(2, max_bit);

		while (next < max_bit)
		{
			//		if (help.isNodeAlive(successor.first.first, successor.first.second) == false)
			//			return;

			long long int newId = currentNode.id + pow(2, next );
			newId = newId % mod;
			nodeInfo node = find_successor(newId);
			if (node.ip == "" || node.id == -1 || node.port == -1)
				break;
			fingerTable[next] = node;
			//fingerTable.push_back(node);
			next++;
		}
	}
void put(string key, string value)
	{
    	if (key == "" || value == "")
    	{
       		 cout << "Key or value field empty\n";
        	return;
    	}

    	else
    	{

        ll keyHash = getHashId(key);
        cout << "Key is " << key << " and hash : " << keyHash << endl;

        
        nodeInfo node2 = find_successor(keyHash);

       

        string ip = node2.ip;
        int port = node2.port;

        struct sockaddr_in connect_server;
        socklen_t l = sizeof(connect_server);

        connect_server.sin_family = AF_INET;
        connect_server.sin_addr.s_addr = inet_addr(ip.c_str());
        connect_server.sin_port = htons(port);


        int sock = socket(AF_INET, SOCK_DGRAM, 0);

        if (sock < 0)
        {
            perror("error");
            exit(-1);
        }

        

        string ipAndPort = "put ";
        int i = 0;
        string hash_string = to_string(keyHash);
        for (i = 0; i < hash_string.size(); i++)
        {
            ipAndPort += hash_string[i];
        }

        ipAndPort += ':';
        for (i = 0; i < value.size(); i++)
        {
            ipAndPort += value[i];
        }

        char key_value_array[100];
        strcpy(key_value_array, ipAndPort.c_str());

        sendto(sock, key_value_array, strlen(key_value_array), 0, (struct sockaddr *)&connect_server, l);

        close(sock);

        cout << "key entered successfully\n";
    }
}
nodeInfo closest_preceding_node(ll nodeId)
{
	ll greatest_smaller,gsi,greatest,gi;
	greatest_smaller=-1;
	gsi=-1;
	greatest=successor.id;
	gi=0;
	for(int i=0;i<max_bit;i++)
	{
		if(fingerTable[i].id<nodeId && fingerTable[i].id>greatest_smaller && fingerTable[i].id!=currentNode.id)
		{
			greatest_smaller=fingerTable[i].id;
			gsi=i;
		}
		if(fingerTable[i].id>greatest )
		{
			greatest=fingerTable[i].id;
			gi=i;
		}
	}
	if(gsi!=-1)
		return fingerTable[gsi];
	else
		return fingerTable[gi];
}

};
void interface(Node node)
	{
		int ch=0;
		while(true)
		{
			cout<<"1.Create ring\n";
			cout<<"2.Join ring\n";
			cout<<"3.Put\n";
			cout<<"4.get \n";
			cout<<"5.Display\n";
			cout<<"6.Exit\n";
			cout<<"Enter choice:";
			
			cin>>ch;
			if(ch==1)
				node.createRing();
			else if(ch==2)
			{
				cout<<"Enter ip:";
				string ip;
				int port;
				cin>>ip;
				cout<<"Enter port:";
				cin>>port;
				node.joinRing(ip,port);
			}
			// else if(ch==3)
			// 	node.setFingerTable();
			// else if(ch==3)
			// {   string key,value;
			// 	cout<<"Enter Key: ";
			//     cin>>key;
			// 	cout << endl;
			// 	cout<<"Enter Value: ";
			// 	cin>>value;
			// 	cout << endl;
			// 	node.put(key,value);

			// }
			else if(ch==5)
				node.display();
			else if(ch==6)
				break;
		}
	}
int main()
{
	Node ob;
	ob.initializeNode();
	interface(ob);
	return 0;
}