jsPsych.plugins['custom-html-response'] = (function() {

    var plugin = {};
  
    plugin.info = {
      name: 'custom-html-response',
      parameters: {
        stimulus: {
          type: jsPsych.plugins.parameterType.HTML_STRING,
          pretty_name: 'Stimulus',
          default: undefined,
          description: 'The HTML string to be displayed'
        },
        choices: {
          type: jsPsych.plugins.parameterType.KEYCODE,
          pretty_name: 'Choices',
          default: jsPsych.ALL_KEYS,
          description: 'The keys the subject is allowed to press to respond to the stimulus.'
        },
        button_label: {
          type: jsPsych.plugins.parameterType.STRING,
          pretty_name: 'Button label',
          default: 'Continue',
          array: true,
          description: 'The label of the button to display'
        }
      }
    };
  
    plugin.trial = function(display_element, trial) {
  
      // display stimulus
      var html = '<div id="jspsych-custom-html-response-stimulus">'+trial.stimulus+'</div>';
  
      // display buttons
      if (trial.button_label.length > 0) {
        var buttons = [];
        if (Array.isArray(trial.button_label)) {
          buttons = trial.button_label;
        } else {
          buttons.push(trial.button_label);
        }
        for (var i = 0; i < buttons.length; i++) {
          html += '<button class="jspsych-btn" id="jspsych-custom-html-response-button-' + i + '" data-choice="' + i + '">' + buttons[i] + '</button>';
        }
      }
  
      display_element.innerHTML = html;
  
      // start time
      var start_time = performance.now();
  
      // add event listeners to buttons
      for (var i = 0; i < trial.button_label.length; i++) {
        display_element.querySelector('#jspsych-custom-html-response-button-' + i).addEventListener('click', function(e){
          var choice = e.currentTarget.getAttribute('data-choice'); // don't use dataset for jsdom compatibility
          after_response(choice);
        });
      }
  
      // add event listener to keyboard
      var keyboard_listener = jsPsych.pluginAPI.getKeyboardResponse({
        callback_function: after_response,
        valid_responses: trial.choices,
        rt_method: 'performance',
        persist: false,
        allow_held_key: false
      });
  
      // function to handle responses by the subject
      function after_response(choice) {
  
        // measure rt
        var end_time = performance.now();
        var rt = end_time - start_time;
  
        // kill any remaining setTimeout handlers
        jsPsych.pluginAPI.clearAllTimeouts();
  
        // kill keyboard listeners
        jsPsych.pluginAPI.cancelAllKeyboardResponses();
  
        // gather the data to store for the trial
        var trial_data = {
          "rt": rt,
          "stimulus": trial.stimulus,
          "response": choice
        };
  
        // clear the display
        display_element.innerHTML = '';
  
        // move on to the next trial
        jsPsych.finishTrial(trial_data);
      }
    };
  
    return plugin;
  })();
  