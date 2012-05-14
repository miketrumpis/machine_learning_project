from distutils.core import setup

if __name__=='__main__':
    setup(
        name = 'recog',
        version = '1.0',
        packages = [
            'recog',
            'recog.conf',
            'recog.faces',
            'recog.dict',
            'recog.opt',
            'recog.support'
            ],
        ext_modules = [],
        package_data = {'recog.conf': ['conf.txt']},
    )
