qwerty_layout = ("`1234567890-=qwertyuiop[]\\asdfghjkl;'zxcvbnm,./"
                 "~!@#$%^&*()_+QWERTYUIOP{}|ASDFGHJKL:\"ZXCVBNM<>?")
 
dvorak_layout = ("`1234567890[]',.pyfgcrl/=\\aoeuidhtns-;qjkxbmwvz"
                 "~!@#$%^&*(){}\"<>PYFGCRL?+|AOEUIDHTNS_:QJKXBMWVZ")

q2d_dict = dict(zip(qwerty_layout, dvorak_layout))
d2q_dict = dict(zip(dvorak_layout, qwerty_layout))
 

def qwerty_to_dvorak(text):
    return ''.join(
        q2d_dict[char] if char in q2d_dict else char
        for char in text)


def dvorak_to_qwerty(text):
    return ''.join(
        d2q_dict[char] if char in d2q_dict else char
        for char in text)
