from collections import OrderedDict

from lookout.style.format.classes import CLASS_INDEX, CLS_DOUBLE_QUOTE, CLS_NEWLINE, CLS_NOOP, \
    CLS_SINGLE_QUOTE, CLS_SPACE, CLS_SPACE_DEC, CLS_SPACE_INC

composite_to_label_names = [
    (CLS_SPACE,),
    (CLS_SPACE, CLS_SINGLE_QUOTE),
    (CLS_SPACE, CLS_DOUBLE_QUOTE),
    (CLS_NEWLINE, ),
    (CLS_NEWLINE, CLS_NEWLINE),
    (CLS_NEWLINE, CLS_SPACE_INC),
    (CLS_NEWLINE, CLS_SPACE_DEC),
    (CLS_NEWLINE, CLS_SPACE_INC, CLS_SPACE_INC),
    (CLS_NEWLINE, CLS_SPACE_DEC, CLS_SPACE_DEC),
    (CLS_NEWLINE, CLS_SPACE_DEC, CLS_SPACE_DEC, CLS_SPACE_DEC),
    (CLS_NEWLINE, CLS_SPACE_DEC, CLS_SPACE_DEC, CLS_SPACE_DEC, CLS_SPACE_DEC),
    (CLS_NEWLINE, CLS_SPACE_DEC, CLS_SPACE_DEC, CLS_SPACE_DEC, CLS_SPACE_DEC, CLS_SPACE_DEC,
        CLS_SPACE_DEC),
    (CLS_SPACE_INC, ),
    (CLS_SPACE_DEC, ),
    (CLS_SINGLE_QUOTE, ),
    (CLS_DOUBLE_QUOTE, ),
    (CLS_NOOP, ),
]

label_composites = [
    tuple(CLASS_INDEX[cls] for cls in label_name)
    for label_name in composite_to_label_names]

labels_to_composite = {x: i for i, x in enumerate(composite_to_label_names)}

original_code = """import { makeToast } from '../../common/app/Toasts/redux';

export default function flashToToast(flash) {
  return Object.keys(flash)
    .map(key => {
      const messages = flash[key];
      return messages.map(message => ({
        message: message.msg,
        type: key,
        timeout: 5000
      }));
    })
    .reduce((toasts, messages) => toasts.concat(messages), [])
    .map(makeToast)
    .map(({ payload }) => payload);
}
"""

# Virtual node index, new target (y) value, result, result for change_locally=True
# if results are equal last can be omitted
cases = OrderedDict([
    ("nothing changed", (
        tuple(),
        tuple(),
        original_code
    )),
    ("remove new line in the end of 4th line", (
        (22,),
        (labels_to_composite[(CLS_NOOP, )],),
        """import { makeToast } from '../../common/app/Toasts/redux';

export default function flashToToast(flash) {
  return Object.keys(flash).map(key => {
    const messages = flash[key];
    return messages.map(message => ({
      message: message.msg,
      type: key,
      timeout: 5000
    }));
  })
  .reduce((toasts, messages) => toasts.concat(messages), [])
  .map(makeToast)
  .map(({ payload }) => payload);
}
""",
        """import { makeToast } from '../../common/app/Toasts/redux';

export default function flashToToast(flash) {
  return Object.keys(flash).map(key => {
      const messages = flash[key];
      return messages.map(message => ({
        message: message.msg,
        type: key,
        timeout: 5000
      }));
    })
    .reduce((toasts, messages) => toasts.concat(messages), [])
    .map(makeToast)
    .map(({ payload }) => payload);
}
""")),
    ("indentation in the beginning", (
        (0,),
        (labels_to_composite[(CLS_SPACE_INC, )],),
        """ import { makeToast } from '../../common/app/Toasts/redux';

 export default function flashToToast(flash) {
   return Object.keys(flash)
     .map(key => {
       const messages = flash[key];
       return messages.map(message => ({
         message: message.msg,
         type: key,
         timeout: 5000
       }));
     })
     .reduce((toasts, messages) => toasts.concat(messages), [])
     .map(makeToast)
     .map(({ payload }) => payload);
 }
""",
        """ import { makeToast } from '../../common/app/Toasts/redux';

export default function flashToToast(flash) {
  return Object.keys(flash)
    .map(key => {
      const messages = flash[key];
      return messages.map(message => ({
        message: message.msg,
        type: key,
        timeout: 5000
      }));
    })
    .reduce((toasts, messages) => toasts.concat(messages), [])
    .map(makeToast)
    .map(({ payload }) => payload);
}
""")),
    ("remove indentation in the 4th line till the end", (
        (15, 114),
        (labels_to_composite[(CLS_NEWLINE, CLS_SPACE_INC)],
         labels_to_composite[(CLS_NEWLINE, CLS_SPACE_DEC, CLS_SPACE_DEC, CLS_SPACE_DEC)]),
        """import { makeToast } from '../../common/app/Toasts/redux';

export default function flashToToast(flash) {
 return Object.keys(flash)
   .map(key => {
     const messages = flash[key];
     return messages.map(message => ({
       message: message.msg,
       type: key,
       timeout: 5000
     }));
   })
   .reduce((toasts, messages) => toasts.concat(messages), [])
   .map(makeToast)
   .map(({ payload }) => payload);
}
""",
        """import { makeToast } from '../../common/app/Toasts/redux';

export default function flashToToast(flash) {
 return Object.keys(flash)
    .map(key => {
      const messages = flash[key];
      return messages.map(message => ({
        message: message.msg,
        type: key,
        timeout: 5000
      }));
    })
    .reduce((toasts, messages) => toasts.concat(messages), [])
    .map(makeToast)
    .map(({ payload }) => payload);
 }
""")),
    ("new line between 6th and 7th regular code lines", (
        (39,),
        (labels_to_composite[(CLS_NEWLINE, CLS_NEWLINE)], ),
        """import { makeToast } from '../../common/app/Toasts/redux';

export default function flashToToast(flash) {
  return Object.keys(flash)
    .map(key => {
      const messages = flash[key];

      return messages.map(message => ({
        message: message.msg,
        type: key,
        timeout: 5000
      }));
    })
    .reduce((toasts, messages) => toasts.concat(messages), [])
    .map(makeToast)
    .map(({ payload }) => payload);
}
""")),
    ("new line in the middle of the 7th code line with indentation increase", (
        (42, 71),
        (labels_to_composite[(CLS_NEWLINE, CLS_SPACE_INC, CLS_SPACE_INC)],
         labels_to_composite[(CLS_NEWLINE, CLS_SPACE_DEC, CLS_SPACE_DEC,
                              CLS_SPACE_DEC, CLS_SPACE_DEC)]),
        """import { makeToast } from '../../common/app/Toasts/redux';

export default function flashToToast(flash) {
  return Object.keys(flash)
    .map(key => {
      const messages = flash[key];
      return messages
        .map(message => ({
          message: message.msg,
          type: key,
          timeout: 5000
        }));
    })
    .reduce((toasts, messages) => toasts.concat(messages), [])
    .map(makeToast)
    .map(({ payload }) => payload);
}
""",
        """import { makeToast } from '../../common/app/Toasts/redux';

export default function flashToToast(flash) {
  return Object.keys(flash)
    .map(key => {
      const messages = flash[key];
      return messages
        .map(message => ({
        message: message.msg,
        type: key,
        timeout: 5000
      }));
  })
    .reduce((toasts, messages) => toasts.concat(messages), [])
    .map(makeToast)
    .map(({ payload }) => payload);
}
""")),
    ("new line in the middle of the 7th code line with indentation decrease", (
        (42, 71),
        (labels_to_composite[(CLS_NEWLINE, CLS_SPACE_DEC, CLS_SPACE_DEC)],
         labels_to_composite[(CLS_NEWLINE, )]),
        """import { makeToast } from '../../common/app/Toasts/redux';

export default function flashToToast(flash) {
  return Object.keys(flash)
    .map(key => {
      const messages = flash[key];
      return messages
    .map(message => ({
      message: message.msg,
      type: key,
      timeout: 5000
    }));
    })
    .reduce((toasts, messages) => toasts.concat(messages), [])
    .map(makeToast)
    .map(({ payload }) => payload);
}
""",
        """import { makeToast } from '../../common/app/Toasts/redux';

export default function flashToToast(flash) {
  return Object.keys(flash)
    .map(key => {
      const messages = flash[key];
      return messages
    .map(message => ({
        message: message.msg,
        type: key,
        timeout: 5000
      }));
      })
    .reduce((toasts, messages) => toasts.concat(messages), [])
    .map(makeToast)
    .map(({ payload }) => payload);
}
""")),
    ("new line in the middle of the 7th code line without indentation increase", (
        (42,),
        (labels_to_composite[(CLS_NEWLINE,)], ),
        """import { makeToast } from '../../common/app/Toasts/redux';

export default function flashToToast(flash) {
  return Object.keys(flash)
    .map(key => {
      const messages = flash[key];
      return messages
      .map(message => ({
        message: message.msg,
        type: key,
        timeout: 5000
      }));
    })
    .reduce((toasts, messages) => toasts.concat(messages), [])
    .map(makeToast)
    .map(({ payload }) => payload);
}
""",
        """import { makeToast } from '../../common/app/Toasts/redux';

export default function flashToToast(flash) {
  return Object.keys(flash)
    .map(key => {
      const messages = flash[key];
      return messages
      .map(message => ({
        message: message.msg,
        type: key,
        timeout: 5000
      }));
    })
    .reduce((toasts, messages) => toasts.concat(messages), [])
    .map(makeToast)
    .map(({ payload }) => payload);
}
""")),
    ("change quotes", (
        (5, 6),
        (labels_to_composite[(CLS_SPACE, CLS_DOUBLE_QUOTE)],
         labels_to_composite[(CLS_DOUBLE_QUOTE,)]),
        """import { makeToast } from "../../common/app/Toasts/redux";

export default function flashToToast(flash) {
  return Object.keys(flash)
    .map(key => {
      const messages = flash[key];
      return messages.map(message => ({
        message: message.msg,
        type: key,
        timeout: 5000
      }));
    })
    .reduce((toasts, messages) => toasts.concat(messages), [])
    .map(makeToast)
    .map(({ payload }) => payload);
}
""")),
    ("remove indentation decrease 11th line", (
        (66,),
        (labels_to_composite[(CLS_NEWLINE,)],),
        """import { makeToast } from '../../common/app/Toasts/redux';

export default function flashToToast(flash) {
  return Object.keys(flash)
    .map(key => {
      const messages = flash[key];
      return messages.map(message => ({
        message: message.msg,
        type: key,
        timeout: 5000
        }));
      })
      .reduce((toasts, messages) => toasts.concat(messages), [])
      .map(makeToast)
      .map(({ payload }) => payload);
  }
""",
        """import { makeToast } from '../../common/app/Toasts/redux';

export default function flashToToast(flash) {
  return Object.keys(flash)
    .map(key => {
      const messages = flash[key];
      return messages.map(message => ({
        message: message.msg,
        type: key,
        timeout: 5000
        }));
    })
    .reduce((toasts, messages) => toasts.concat(messages), [])
    .map(makeToast)
    .map(({ payload }) => payload);
}
"""
    )),
    ("change indentation decrease to indentation increase 11th line", (
        (66,),
        (labels_to_composite[(CLS_NEWLINE, CLS_SPACE_INC, CLS_SPACE_INC)],),
        """import { makeToast } from '../../common/app/Toasts/redux';

export default function flashToToast(flash) {
  return Object.keys(flash)
    .map(key => {
      const messages = flash[key];
      return messages.map(message => ({
        message: message.msg,
        type: key,
        timeout: 5000
          }));
        })
        .reduce((toasts, messages) => toasts.concat(messages), [])
        .map(makeToast)
        .map(({ payload }) => payload);
    }
""",
        """import { makeToast } from '../../common/app/Toasts/redux';

export default function flashToToast(flash) {
  return Object.keys(flash)
    .map(key => {
      const messages = flash[key];
      return messages.map(message => ({
        message: message.msg,
        type: key,
        timeout: 5000
          }));
    })
    .reduce((toasts, messages) => toasts.concat(messages), [])
    .map(makeToast)
    .map(({ payload }) => payload);
}
"""
    )),
    ("change indentation decrease to indentation increase 11th line but keep the rest", (
        (66, 71),
        (labels_to_composite[(CLS_NEWLINE, CLS_SPACE_INC, CLS_SPACE_INC)],
         labels_to_composite[(CLS_NEWLINE, CLS_SPACE_DEC, CLS_SPACE_DEC, CLS_SPACE_DEC,
                              CLS_SPACE_DEC, CLS_SPACE_DEC, CLS_SPACE_DEC)],),
        """import { makeToast } from '../../common/app/Toasts/redux';

export default function flashToToast(flash) {
  return Object.keys(flash)
    .map(key => {
      const messages = flash[key];
      return messages.map(message => ({
        message: message.msg,
        type: key,
        timeout: 5000
          }));
    })
    .reduce((toasts, messages) => toasts.concat(messages), [])
    .map(makeToast)
    .map(({ payload }) => payload);
}
""",
        """import { makeToast } from '../../common/app/Toasts/redux';

export default function flashToToast(flash) {
  return Object.keys(flash)
    .map(key => {
      const messages = flash[key];
      return messages.map(message => ({
        message: message.msg,
        type: key,
        timeout: 5000
          }));
})
    .reduce((toasts, messages) => toasts.concat(messages), [])
    .map(makeToast)
    .map(({ payload }) => payload);
}
"""
    ))
])


case_template_to_copy_paste = \
    ("Name", (
        (-1, -1),
        (labels_to_composite[(CLS_NOOP,)],
         labels_to_composite[(CLS_NOOP,)]),
        """import { makeToast } from '../../common/app/Toasts/redux';

export default function flashToToast(flash) {
  return Object.keys(flash)
    .map(key => {
      const messages = flash[key];
      return messages.map(message => ({
        message: message.msg,
        type: key,
        timeout: 5000
      }));
    })
    .reduce((toasts, messages) => toasts.concat(messages), [])
    .map(makeToast)
    .map(({ payload }) => payload);
}
"""))
