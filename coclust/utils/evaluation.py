import numpy as np
from collections import Counter

def compute_nmi(predicted_labels, true_labels) :
    n=len(predicted_labels)
    if n != len(true_labels) :
        print "Bad list lengths!"
        return # Gerer exception
    pairs=zip(predicted_labels, true_labels)
    n=float(n)
    x_cnt=Counter(predicted_labels)
    y_cnt=Counter(true_labels)
    xy_cnt = Counter(pairs)

    x_prob= { x : x_cnt[x] / n for x in x_cnt.keys() }
    y_prob= {y :  y_cnt[y] / n for y in y_cnt.keys() }
    xy_prob = { (x,y) : xy_cnt[x,y] / n for (x,y) in xy_cnt.keys() }

#    print "predicted"
#    print predicted_labels
#    print "true"
#    print true_labels

##    for x in x_prob :
##        print "p_x({})={:.3f}".format(x,x_prob[x])
##    for y in y_prob :
##        print "p_y({})={:.3f}".format(y,y_prob[y])
##    for (x,y) in xy_prob :
##        print "p_xy({},{})={:.3f}".format(x,y,xy_prob[x,y])
##
##    for (x,y) in xy_prob :
##        print x, y ,np.log2( xy_prob[x,y] / (x_prob[x] *  y_prob[y]))
##


    mi=0.
    for (x,y) in xy_prob :
        mi+= xy_prob[x,y] *  np.log2( xy_prob[x,y] / (x_prob[x] *  y_prob[y]) )
    print "MI:" ,mi
    # MI is  not  NMI







#compute_nmi([1,0,0,2,2] , [0,1,1,2,2])
#compute_nmi([1,0,0,2,0,0,2] , [0,1,1,2,1,1,2])
