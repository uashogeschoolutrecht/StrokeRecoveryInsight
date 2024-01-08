import numpy as np
'''
Is used to calculate the asymmetry in the walking signal. 
The first 5 and last 5 strides are alwyas skipped.
The first 10 and last 10 steps are alwyas skipped.

Patterson, K. K., Gage, W. H., Brooks, D., Black, S. E., & McIlroy, W. E. 
(2010). Evaluation of gait symmetry after stroke: A comparison of current
 methods and recommendations for standardization. Gait & Posture, 31(2), 
 241–246. doi:10.1016/j.gaitpost.2009.10.014 

'''


class Asymmetry:
    def __init__(self, leftFoot, rightFoot, lowBack, pareticFoot, 
                 printje = None):
        '''
        This script is used to calculate the symmetryIndex, GaitAsymmetry and
        SymmetryAngle.
        The required input is the stride times, standphases, swingphases, and 
        peaks of the leftfoot, rightfoot and lowback. 
        The paretic foot (Left, Right, None) is required to do the calculations. 
        If no foot is given the left foot will be used.
        Additionally to normalise the variables we require person length in cm
        . 
        '''
        
        def meanPar_NonPar(par, nonpar, begin, end):
            outpar = np.array([])
            outnonpar = np.array([])
            
            for i in par[begin:-end]:
                outpar = np.concatenate((outpar, np.array([len(i)])))
        
            for i in nonpar[begin:-end]:
                outnonpar = np.concatenate((outnonpar, np.array([len(i)])))
            
            return outpar, outnonpar
        
        def symmetryIndex(par, nonpar):
           SI = ((par / nonpar ) / (0.5 * (par + nonpar)))
           return SI

        def GaitAsymmetry(par, nonpar):
            GA = 100 * (np.log(par/ nonpar))
            return GA
        
        def SymmetryAngle(par, nonpar):
            SA = (45 - np.arctan(par/nonpar)) / 90
            return SA

          
        if pareticFoot == 'Left': 
            parFoot     = leftFoot
            NonParFoot  = rightFoot
        elif pareticFoot == 'Right':
            parFoot     = rightFoot
            NonParFoot  = leftFoot    
        else:
            parFoot     = leftFoot
            NonParFoot  = rightFoot  

        # Standphase asymmetry

        standPar, standNONPar = meanPar_NonPar(parFoot.standphases, 
                                                       NonParFoot.standphases, 
                                                       5, 
                                                       5)
        meanstandPar = np.mean(standPar) * parFoot.sampleFreq
        meanstandNONPar = np.mean(standNONPar) * parFoot.sampleFreq
        self.SRstandphase   = (meanstandPar / 
                               meanstandNONPar)     
        
        # Swingphase asymetry
        swingPar, swingNONPar = meanPar_NonPar(parFoot.swingphases, 
                                                       NonParFoot.swingphases, 
                                                       5, 
                                                       5)
        meanswingPar = np.mean(swingPar)* parFoot.sampleFreq
        meanSwingNONPar = np.mean(swingNONPar)* parFoot.sampleFreq  
        self.SRswingphase   = (meanswingPar/  meanSwingNONPar)     
        
        # Swingphase/standphase asymetry
        shortest =    np.argmin([len(swingPar),len(swingNONPar),
                                 len(standPar),len(standNONPar)])
        if shortest == 0:
            inputlenth = len(swingPar) -1 
        elif shortest == 1:
            inputlenth = len(swingNONPar)-1
        elif shortest == 2:
            inputlenth = len(standPar)-1
        else:
            inputlenth = len(standNONPar)   -1
        
        swing_stancePar = swingPar[inputlenth] / standPar[inputlenth]
        swing_stanceNONPar = swingNONPar[inputlenth] / standNONPar[inputlenth]
        meanswing_stancePar = np.mean(swing_stancePar)* parFoot.sampleFreq
        meanswing_stanceNONPar = np.mean(swing_stanceNONPar)* parFoot.sampleFreq
        self.SRswing_stance  = (meanswing_stancePar/  meanswing_stanceNONPar)     

        #Symmetry index
        self.SIswing_stancePar = symmetryIndex(meanswing_stancePar, meanswing_stanceNONPar)                  
        self.SIstandphase = symmetryIndex(meanstandPar, meanstandNONPar)                  
        self.SIswingphase = symmetryIndex(meanswingPar,meanSwingNONPar)                 
                    
        # Gait asymmetry
        self.GAswing_stancePar = GaitAsymmetry(meanswing_stancePar, meanswing_stanceNONPar)                  
        self.GAstandphase = GaitAsymmetry(meanstandPar, meanstandNONPar)                  
        self.GAswingphase = GaitAsymmetry(meanswingPar,meanSwingNONPar)                 
       
        # Symmetry Angle 
        self.SAswing_stancePar = SymmetryAngle(meanswing_stancePar, meanswing_stanceNONPar)                  
        self.SAstandphase = SymmetryAngle(meanstandPar, meanstandNONPar)                  
        self.SAswingphase = SymmetryAngle(meanswingPar,meanSwingNONPar)                 
            
                # Stride peak asymmetry
        accelerationPar = parFoot.acceleration[:, 2]
        accelerationNonPar = NonParFoot.acceleration[:, 2]
        
        peaksPar = accelerationPar[parFoot.peaksVT]
        peaksNonPar = accelerationNonPar[NonParFoot.peaksVT]


        if len(peaksPar ) > len(peaksNonPar):
            peakDiff = (peaksPar[0:len(peaksNonPar)] -
                              peaksNonPar)
        elif len(peaksNonPar ) > len(peaksPar):
            peakDiff = (peaksPar -
                              peaksNonPar[0:len(peaksPar)])
        else:
            peakDiff = (peaksPar -peaksNonPar)
        self.Amplitudeasym = np.mean(peakDiff)
        self.AmplitudeSTDasym = np.std(peakDiff)
        
        
        # Low Back peak asymetry
        firstFoot = lowBack.direction
        acceleration = lowBack.acceleration[lowBack.walkSig,0]
        
        if firstFoot == 'Right':
            lowBackLeft = lowBack.peaks[1::2]
            lowBackRight = lowBack.peaks[0::2]
        else:
            lowBackLeft = lowBack.peaks[0::2]
            lowBackRight = lowBack.peaks[1::2]
            
        peakslowBackLeft= acceleration[lowBackLeft]
        peakslowBackRight = acceleration[lowBackRight]
        
        
        if pareticFoot == 'Left':
            parFoot     = peakslowBackLeft
            NonParFoot  = peakslowBackRight
        elif pareticFoot == 'Right':
            parFoot     = peakslowBackRight
            NonParFoot  = peakslowBackLeft    
        else:
            parFoot     = peakslowBackLeft
            NonParFoot  = peakslowBackRight
            
        if len(parFoot ) > len(NonParFoot):
            peakDiff = (parFoot[0:len(NonParFoot)  ] -
                              NonParFoot)
        elif len(NonParFoot ) > len(parFoot):
            peakDiff = (parFoot -
                              NonParFoot[0:len(parFoot)  ])
        else:
            peakDiff = (parFoot -NonParFoot)

        self.stepPeakDiffSTD = np.std(peakDiff)
        self.lowBackPeakDiffValues = (np.mean(peakslowBackLeft) / 
                   np.mean(peakslowBackRight))
        
      
        
        if printje:    
            print('SRstandphase ', self.SRstandphase)
            print('SRswingphase',self.SRswingphase)
            print('SRswing_stance',self.SRswing_stance) 
            print('SIstandphase  '  , self.SIstandphase)             
            print('SIswingphase   '   ,self.SIswingphase) 
            print('SIswing_stancePar   '   ,self.SIswing_stancePar) 
            print('GAstandphase     ',self.GAstandphase)            
            print('GAswingphase  ', self.GAswingphase)
            print('GAswing_stancePar  ', self.GAswing_stancePar) 
            print('SAstandphase    '  ,self.SAstandphase)           
            print('SAswingphase     '   , self.SAswingphase)  
            print('SAswing_stancePar     '   , self.SAswing_stancePar)  
            print('stepPeakDiffSTD' , self.stepPeakDiffSTD )
            print('lowBackPeakDiffValues',  self.lowBackPeakDiffValues)
            print('Amplitudeasym', self.Amplitudeasym)
            print('AmplitudeSTDasym',self.AmplitudeSTDasym)
            
            
    