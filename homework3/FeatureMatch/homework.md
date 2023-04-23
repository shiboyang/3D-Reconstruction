# Feature Match

作业说明文档：

在这次作业中，您需要完成以下任务：

1、使用OpenCV提取SIFT特征并进行特征匹配。  
2、在RANSAC算法中，选择8个匹配点，并完成Essential matrix的计算。  
3、计算内点数量。  
4、如果当前内点数量大于最大内点数量，则更新最佳Essential Matrix。  

作业要求：

请补充代码，提取SIFT特征并进行特征匹配。使用OpenCV的SIFT特征检测器检测两幅图像中的关键点，提取描述子并使用Brute-Force Matcher进行匹配。最后，将匹配点的坐标存储在pts1和pts2中。  

在RANSAC算法的每次迭代中，从pts1和pts2中随机选择8个匹配点。实现从这8个匹配点计算Essential matrix的功能，可以使用OpenCV的findEssentialMat函数来完成这个任务。  

计算内点数量。对于每个匹配点对，计算它们是否满足Essential matrix的约束。如果满足约束，则将其标记为内点。

如果当前内点数量大于最大内点数量，则更新最佳Essential Matrix。在RANSAC算法的每次迭代中，需要检查当前内点数量是否大于最大内点数量。如果大于，则将当前的Essential matrix作为最佳Essential matrix，并更新最大内点数量。

作业代码编译运行方式：

    编译
    mkdir build
    cd build
    cmake ..
    make 
    
    运行
    ./FeatureMatch  