## this file used to compute some measurement 
import sys
import os

import numpy as np
import openmesh as om
try:
    import pyigl as igl
except:
    import src.pyigl as igl
# rigid registeration of 2 meshes
def rigid_registeration(src_array, tar_array, index):
	'''
	apple rigid registeration between 2 meshes.
	'''
	if index is not None:
		src_array = src_array[index]
		tar_array = tar_array[index]
	M = np.matmul((src_array - np.mean(src_array, axis=0)).T, (tar_array - np.mean(tar_array, axis=0)))
	u,s,v= np.linalg.svd(M)
	# print(np.dot(np.dot(u,np.diag(s)), v))
	sig = np.ones(s.shape)
	sig[-1] = np.linalg.det(np.dot(u,v))
	R = np.matmul(np.matmul(v.T, np.diag(sig)), u.T)
	t = np.mean(tar_array, axis=0) - np.matmul(R,np.mean(src_array, axis=0))
	return R, t

def compute_distance(src, tar, index=None):
	'''
	compute the L2 distance of average distance between 2 meshes on each vertice
	'''
	src_mesh = om.read_trimesh(src)
	tar_mesh = om.read_trimesh(tar)
	point_array_src = src_mesh.points()
	point_array_tar = tar_mesh.points()
	R, t = rigid_registeration(point_array_src, point_array_tar, index)
	register_src = np.dot(R, point_array_src.T).T + np.tile(t,point_array_src.shape[0]).reshape(-1,3)
	# distance = np.mean(np.square(point_array_tar - register_src ))\

 	## new way for front face distance computation
	diff_array = (point_array_tar-register_src)[index]
	print(np.shape(diff_array))
	distance_mean = np.mean(np.sqrt(np.sum(np.square(diff_array),axis=1)))
	distance_max = np.max(np.sqrt(np.sum(np.square(diff_array),axis=1)))
	distance_no_rigid = np.mean(np.sqrt(np.sum(np.square(diff_array),axis=1)))
	return distance_mean, distance_max,distance_no_rigid

def compute_variance(data_array):
	'''
	compute variance of data_array, Var(x) = sqrt(E(x-E(x))^2))
	data_array size: feature_num * feature_dim 
	'''
	return np.mean(np.std(data_array, axis=0))


def write_align_mesh(src, tar, filename, index = None):
	src_mesh=om.read_trimesh(src)
	tar_mesh=om.read_trimesh(tar)
	point_array_src = src_mesh.points()
	point_array_tar = tar_mesh.points()
	R, t = rigid_registeration(point_array_src, point_array_tar, index)
	register_array_src = np.dot(R, point_array_src.T).T + np.tile(t,point_array_src.shape[0]).reshape(-1,3)
	new_V = igl.eigen.MatrixXd(register_array_src.astype(np.float64))
	V = igl.eigen.MatrixXd()
	F = igl.eigen.MatrixXi()
	igl.readOBJ(src, V,F)
	igl.writeOBJ(filename, new_V, F)
	# om.write_mesh(filename, src_mesh)

def cal_distance_in_file(src_file_format, tar_file_format, vis = False):
    avg_dis=[]
    index=None
    for i in range(141,151):
        for j in range(20):
            if j in []:
                continue
            dis,_,_ = compute_distance(src_file_format.format(i, j), tar_file_format.format(i, j), index)
            avg_dis.append(dis)
            if vis:
                print('Loading Mesh: {}, Expression: {}, vertice distacnce: {}'.format(i,j, dis))

    
    print('Avg distance: {}'.format(np.mean(avg_dis)))
    print('extreme differ: {}'.format(np.max(np.abs(avg_dis - np.mean(avg_dis)))))
    print('median distance: {}'.format(np.median(avg_dis)))
    return avg_dis

def cal_id_disentanglement_in_file(file_format, use_registeration = True, vis = True):
    index = None
    loss_log = []
    for i in range(141,151):
        data_array=[]
        for j in range(0, 20):
            mesh=om.read_trimesh(file_format.format(i,j))
            point_array=mesh.points()
            if use_registeration:
                if j > 0:
                    R, t = rigid_registeration(point_array, fix, index)
                    register_src = np.dot(R, point_array.T).T + np.tile(t,point_array.shape[0]).reshape(-1,3)
                    point_array = register_src
                else:
                    fix = point_array
            data_array.append(point_array.reshape(1,-1))
        data_array=np.concatenate(data_array,axis=0)
        loss_log.append(compute_variance(data_array))
        if vis:
            print('Exp: {}, var: {}'.format(i, loss_log[-1]))
    print('our id Average variance: {}'.format(np.mean(loss_log)))
    print('our id Median variance: {}'.format(np.median(loss_log)))
    print('extreme var: {}'.format(max(np.max(loss_log)-np.mean(loss_log), np.mean(loss_log)- np.min(loss_log))))
    return loss_log

def cal_exp_disentanglement_in_file(file_format, use_registeration = True, vis = True):
    index = None
    loss_log = []
    for j in range(0, 20):
        data_array=[]
        for i in range(141,151):
            mesh=om.read_trimesh(file_format.format(i,j))
            point_array=mesh.points()
            if use_registeration:
                if j > 0:
                    R, t = rigid_registeration(point_array, fix, index)
                    register_src = np.dot(R, point_array.T).T + np.tile(t,point_array.shape[0]).reshape(-1,3)
                    point_array = register_src
                else:
                    fix = point_array
            data_array.append(point_array.reshape(1,-1))
        data_array=np.concatenate(data_array,axis=0)
        loss_log.append(compute_variance(data_array))
        if vis:
            print('Exp: {}, var: {}'.format(j, loss_log[-1]))
    print('our exp Average variance: {}'.format(np.mean(loss_log)))
    print('our exp Median variance: {}'.format(np.median(loss_log)))
    print('extreme var: {}'.format(max(np.max(loss_log)-np.mean(loss_log), np.mean(loss_log)- np.min(loss_log))))
    return loss_log

if __name__ == '__main__':
    
    # index=np.loadtxt('index_68_jl.txt', dtype=int)
    # data_root='/raid/jzh/FWH_1018/to_be_aligned'
    # for i in range(141,151):
    #     os.makedirs(os.path.join('Tester_'+str(i),'AlignPose'))
    #     for j in range(0,20):
    #         src_path=os.path.join(data_root,'Tester_'+str(i),'TrainingPose','pose_'+str(j)+'.obj')
    #         mean_path='/raid/jzh/FeatureDistangle/data/distangle/Mean_Face.obj'
    #         out_path=os.path.join('Tester_'+str(i),'AlignPose','pose_'+str(j)+'.obj')
    #         write_align_mesh(src_path, mean_path, out_path, index)
#    write_align_mesh('/raid/jzh/FWH_1018/to_be_aligned/Tester_141/TrainingPose/pose_1.obj','/raid/jzh/FeatureDistangle/data/distangle/Mean_Face.obj', 'align_test.obj', index)
    
    #cal_distance_in_file('/raid/jzh/alignpose/Tester_{}/AlignPose/pose_{}.obj', '/home/jzh/good_test_data/mesh_fcfc+xyz/rec_plus/Feature{}/{}.obj')
    # --------------    
    # test distance
    # --------------
    avg_dis=[]
    count=0
    meshpath_root = '/raid/jzh/alignpose'
    test_root = '/raid/jzh/SVD_TestData/Reconstruct_data_trainingpose_1018'#mesh_gcn_fc_fusion_training_pose,mesh_fcfc+,
    #/raid/jzh/SVD_TestData/Reconstruct_data_trainingpose_1018/Tester_141/pose_0_reconstruct.obj
    index = np.loadtxt('front_part_v.txt', dtype=int)
    for i in range(141,151):
        for j in range(20):
            if j in []:
                continue
            tar_path=os.path.join(meshpath_root,'Tester_{}/AlignPose/pose_{}.obj'.format(i,j))
            #src_path=os.path.join(meshpath_root,'Reconstruct_data','Tester_'+str(i),'face_'+str(j)+'_reconstruct.obj')
            src_path=os.path.join(test_root,'Tester_{}/pose_{}_reconstruct.obj'.format(i,j))#'Tester_{}/pose_{}_reconstruct.obj
            #print(src_path)
            #print(tar_path)
            dis,_,_ = compute_distance(src_path, tar_path, index)
            avg_dis.append(dis)
            print('Loading Mesh: {}, Expression: {}, vertice distacnce: {}'.format(i,j, dis))
            count=count+1
    
#    
#    #compute distance 
#    
#    
#    print('Avg distance: {}'.format(np.mean(avg_dis)))
#    print('extreme differ: {}'.format(np.max(np.abs(avg_dis - np.mean(avg_dis)))))
#    print('median distance: {}'.format(np.median(avg_dis)))
    
    
    
#    avg_dis=[]
#    count=0
#    meshpath_root = '/raid/jzh/meshes/mesh_coma_1103'#/raid/jzh/FeatureDistangle/data/mesh/'
#    test_root = '/raid/jzh/meshes/mesh_coma_1103'#/raid/jzh/FeatureDistangle/data/mesh/'
#    
#    index=None
#    for i in range(12):
#        for j in range(1,len(os.listdir(os.path.join(meshpath_root,'ori/Feature{}/'.format(i))))):
#            tar_path=os.path.join(meshpath_root,'ori/Feature{}/{}.obj'.format(i,j))
#            #src_path=os.path.join(meshpath_root,'Reconstruct_data','Tester_'+str(i),'face_'+str(j)+'_reconstruct.obj')
#            src_path=os.path.join(test_root,'rec/Feature{}/{}.obj'.format(i,j))
#            #print(src_path)
#            #print(tar_path)
#            dis,_,_ = compute_distance(src_path, tar_path, index)
#            avg_dis.append(dis)
#            print('Loading Mesh: {}, Expression: {}, vertice distacnce: {}'.format(i,j, dis))
#            count=count+1
#            
#    print('Avg distance: {}'.format(np.mean(avg_dis)))
#    print('extreme var: {}'.format(np.max(np.abs(avg_dis - np.mean(avg_dis)))))

#     #meshpath_root='/home/jzh/Reconstruct_data_1016'
    
    '''
#-----------------------
# test id on training pose (ours fc)
#-----------------------    
    index = None
    #meshpath_root='/home/jzh/good_test_data/mesh_rimd_fcfc_plus/id'#good_test_data/mesh_rimd_fcfc_plus
    meshpath_root ='/raid/jzh/meshes/mesh1023/id'
    #meshpath_root='/raid/jzh/FWH_1018/Reconstruct_data_trainingpose_1018'
    loss_log = []
    for i in range(141,151):
        data_array=[]
        for j in range(0, 47):
            data_path=os.path.join(meshpath_root,'Feature'+str(i),str(j)+'.obj')
            # data_path=os.path.join(meshpath_root,'Tester_'+str(j),'pose_'+str(i)+'_sameid.obj')
            mesh=om.read_trimesh(data_path)
            point_array=mesh.points()#.reshape(1,-1)
            if j > 0:
                R, t = rigid_registeration(point_array, fix, index)
                register_src = np.dot(R, point_array.T).T + np.tile(t,point_array.shape[0]).reshape(-1,3)
                point_array = register_src
            else:
                fix = point_array
            data_array.append(point_array.reshape(1,-1))
        data_array=np.concatenate(data_array,axis=0)
        loss_log.append(compute_variance(data_array))
        print('Exp: {}, var: {}'.format(i, loss_log[-1]))
    print('our id Average variance: {}'.format(np.mean(loss_log)))
    print('our id Median variance: {}'.format(np.median(loss_log)))
    print('extreme var: {}'.format(max(np.max(loss_log)-np.mean(loss_log), np.mean(loss_log)- np.min(loss_log))))
     
     
     
     
     
     
#-----------------------
# test exp on training pose (ours fc)
#-----------------------    
    #meshpath_root='/home/jzh/good_test_data/mesh_rimd_fcfc_plus/exp'
    meshpath_root = '/raid/jzh/meshes/mesh1023/exp'
    loss_log = []
    for i in range(0,47):
        data_array=[]
        for j in range(141, 151):
            data_path=os.path.join(meshpath_root,'Feature'+str(j),str(i)+'.obj')
            mesh=om.read_trimesh(data_path)
            point_array=mesh.points()
            if j > 141:
                R, t = rigid_registeration(point_array, fix, index)
                register_src = np.dot(R, point_array.T).T + np.tile(t,point_array.shape[0]).reshape(-1,3)
                point_array = register_src
            else:
                fix = point_array
            data_array.append(point_array.reshape(1,-1))
        data_array=np.concatenate(data_array,axis=0)
        loss_log.append(compute_variance(data_array))
        print('Exp: {}, var: {}'.format(i, loss_log[-1]))
    print('our exp Average variance: {}'.format(np.mean(loss_log)))
    print('our exp Median variance: {}'.format(np.median(loss_log)))
    print('extreme var: {}'.format(max(np.max(loss_log)-np.mean(loss_log), np.mean(loss_log)- np.min(loss_log))))
    '''
#    
#        exp_id = 0
#    for exp_id in [13,24,30,42,50,81,104,130,132,141]:
#    
#        # index = None
#        index = np.loadtxt('front_part_v.txt', dtype=int)
#        #print(compute_distance('/home/jzh/Reconstruct_data_1016/Tester_{}/face_0_reconstruct.obj'.format(exp_id), '/raid/jzh/align_warehouse_all/Tester_{}/face_0.obj'.format(exp_id), index)[2])
#        print(compute_distance('/raid/jzh/FWH_1112/Reconstruct_data_trainingpose_1112_weakly/Tester_{}/pose_0_reconstruct.obj'.format(exp_id), '/raid/jzh/FWH_1112/AlignTrainingPose_1112/Tester_{}/pose_0.obj'.format(exp_id), index)[0])
        #print(compute_distance('/home/jzh/mesh/id_distangle_{}_aligned.obj'.format(exp_id), '/raid/jzh/align_warehouse_all/Tester_{}/face_0.obj'.format(exp_id), index)[0])
    index = np.loadtxt('front_part_v.txt', dtype=int)
    
    #print(compute_distance('/home/jzh/mesh/result_id.obj', '/home/jzh/mesh/target_id.obj', index)[0])
    #print(compute_distance('/home/jzh/mesh/start_id.obj', '/home/jzh/mesh/target_id.obj', index)[0])
    #print(compute_distance('/raid/jzh/FeatureDistangle/data/mesh/bp_plus/Feature142/1.obj', '/raid/jzh/FeatureDistangle/data/mesh/ori/Feature142/1.obj', index)[0])
#    print(compute_distance('mesh/ori_distangle_{}.obj'.format(exp_id), 'mesh/rec_distangle_{}.obj'.format(exp_id), index)[0])
#    print(compute_distance('mesh/plus_distangle_{}.obj'.format(exp_id), 'mesh/ori_distangle_{}.obj'.format(exp_id), index)[0])
#    print(compute_distance('/raid/jzh/COMA_data/FaceTalk_170725_00137_TA/mouth_down/mouth_down.00000{}_reconstruct.obj'.format(exp_id), '/raid/jzh/COMA_data/FaceTalk_170725_00137_TA/eyebrow/eyebrow.00000{}_reconstruct.obj'.format(exp_id), index)[0])
    for i in range(47):
        print(compute_distance('/raid/jzh/FeatureDistangle/data/mesh/bp_plus/Feature141/{}.obj'.format(i), '/raid/jzh/FeatureDistangle/data/mesh/ori/Feature141/{}.obj'.format(i), index)[0])
    
#    
#    index = None
#    log = 0
#    for i in range(144):
#        logi = compute_distance('mesh/ori_coma_{}.obj'.format(i), 'mesh/plus_coma_{}.obj'.format(i), index)
#        print(logi)
#        log +=logi[0]
#    print(log/144)
