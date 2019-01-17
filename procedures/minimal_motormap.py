import sys
import analyzeMotorMap

'''
From ASIAA data, construct minimal motormaps
'''

def main ( datadir, xml_file ):
    ammt = analyzeMotorMap.AnalyzeMotorMapTask ( datadir, 400 )
    ammt.load_xml ( xml_file, overwrite_centers=True )
    out = ammt.catalog_positions ()
    angles = ammt.fix_geometry ( *out )
    mmaps = ammt.generate_motormap ( *angles )
    return mmaps

#if __name__=='__main__':
#    main ( *sys.argv )
