from .transformer import former_vit_tiny, former_vit_mini, former_vit_small, former_vit_base, former_vit_large
# do not use qkvlution, it adds parameters but does not improve accuracy compared with kv or qvlution
from .translution import qkvlution_vit_tiny, qkvlution_vit_mini, qkvlution_vit_small, qkvlution_vit_base, qkvlution_vit_large
from .translution import kvlution_vit_tiny, kvlution_vit_mini, kvlution_vit_small, kvlution_vit_base, kvlution_vit_large
from .translution import qvlution_vit_tiny, qvlution_vit_mini, qvlution_vit_small, qvlution_vit_base, qvlution_vit_large
from .translution import qklution_vit_tiny, qklution_vit_mini, qklution_vit_small, qklution_vit_base, qklution_vit_large
from .translution import qlution_vit_tiny, qlution_vit_mini, qlution_vit_small, qlution_vit_base, qlution_vit_large
from .translution import klution_vit_tiny, klution_vit_mini, klution_vit_small, klution_vit_base, klution_vit_large
from .translution import vlution_vit_tiny, vlution_vit_mini, vlution_vit_small, vlution_vit_base, vlution_vit_large
# do not use lor_qkvlution, it adds parameters but does not improve accuracy compared with lor_kv or lor_qvlution
from .lor_translution import lor_qkvlution_vit_tiny, lor_qkvlution_vit_mini, lor_qkvlution_vit_small, lor_qkvlution_vit_base, lor_qkvlution_vit_large
from .lor_translution import lor_kvlution_vit_tiny, lor_kvlution_vit_mini, lor_kvlution_vit_small, lor_kvlution_vit_base, lor_kvlution_vit_large
from .lor_translution import lor_qvlution_vit_tiny, lor_qvlution_vit_mini, lor_qvlution_vit_small, lor_qvlution_vit_base, lor_qvlution_vit_large
from .lor_translution import lor_qklution_vit_tiny, lor_qklution_vit_mini, lor_qklution_vit_small, lor_qklution_vit_base, lor_qklution_vit_large
from .lor_translution import lor_qlution_vit_tiny, lor_qlution_vit_mini, lor_qlution_vit_small, lor_qlution_vit_base, lor_qlution_vit_large
from .lor_translution import lor_klution_vit_tiny, lor_klution_vit_mini, lor_klution_vit_small, lor_klution_vit_base, lor_klution_vit_large
from .lor_translution import lor_vlution_vit_tiny, lor_vlution_vit_mini, lor_vlution_vit_small, lor_vlution_vit_base, lor_vlution_vit_large
