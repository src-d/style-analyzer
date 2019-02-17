
;(function ($) {

/*	============================================================
 *	BEGIN WIDGET: $( selector ).layout( {options} );
 *	============================================================
 */
$.fn.layout = function (opts) {
	var

	// local aliases to global data
	browser	= $.layout.browser

,	initResizable = function (panes) {
		var	draggingAvailable = $.layout.plugins.draggable
		,	side // set in start()
		;
		panes = panes ? panes.split(",") : _c.borderPanes;

		$.each(panes, function (idx, pane) {
			var o = options[pane];
			if (!draggingAvailable || !$Ps[pane] || !o.resizable) {
				o.resizable = false;
				return true; // skip to next
			}

			var s		= state[pane]
			,	z		= options.zIndexes
			,	c		= _c[pane]
			,	side	= c.dir=="horz" ? "top" : "left"
			,	$P 		= $Ps[pane]
			,	$R		= $Rs[pane]
			,	base	= o.resizerClass
			,	lastPos	= 0 // used when live-resizing
			,	r, live // set in start because may change
			//	'drag' classes are applied to the ORIGINAL resizer-bar while dragging is in process
			,	resizerClass		= base+"-drag"				// resizer-drag
			,	resizerPaneClass	= base+"-"+pane+"-drag"		// resizer-north-drag
			//	'helper' class is applied to the CLONED resizer-bar while it is being dragged
			,	helperClass			= base+"-dragging"			// resizer-dragging
			,	helperPaneClass		= base+"-"+pane+"-dragging" // resizer-north-dragging
			,	helperLimitClass	= base+"-dragging-limit"	// resizer-drag
			,	helperPaneLimitClass = base+"-"+pane+"-dragging-limit"	// resizer-north-drag
			,	helperClassesSet	= false 					// logic var
			;

			if (!s.isClosed)
				$R.attr("title", o.tips.Resize)
				  .css("cursor", o.resizerCursor); // n-resize, s-resize, etc

			$R.draggable({
				containment:	$N[0] // limit resizing to layout container
			,	axis:			(c.dir=="horz" ? "y" : "x") // limit resizing to horz or vert axis
			,	delay:			0
			,	distance:		1
			,	grid:			o.resizingGrid
			//	basic format for helper - style it using class: .ui-draggable-dragging
			,	helper:			"clone"
			,	opacity:		o.resizerDragOpacity
			,	addClasses:		false // avoid ui-state-disabled class when disabled
			//,	iframeFix:		o.draggableIframeFix // TODO: consider using when bug is fixed
			,	zIndex:			z.resizer_drag

			,	start: function (e, ui) {
					// REFRESH options & state pointers in case we used swapPanes
					o = options[pane];
					s = state[pane];
					// re-read options
					live = o.livePaneResizing;

					// ondrag_start callback - will CANCEL hide if returns false
					// TODO: dragging CANNOT be cancelled like this, so see if there is a way?
					if (false === _runCallbacks("ondrag_start", pane)) return false;

					s.isResizing		= true; // prevent pane from closing while resizing
					state.paneResizing	= pane; // easy to see if ANY pane is resizing
					timer.clear(pane+"_closeSlider"); // just in case already triggered

					// SET RESIZER LIMITS - used in drag()
					setSizeLimits(pane); // update pane/resizer state
					r = s.resizerPosition;
					lastPos = ui.position[ side ]

					$R.addClass( resizerClass +" "+ resizerPaneClass ); // add drag classes
					helperClassesSet = false; // reset logic var - see drag()

					// DISABLE TEXT SELECTION (probably already done by resizer.mouseOver)
					$('body').disableSelection(); 

					// MASK PANES CONTAINING IFRAMES, APPLETS OR OTHER TROUBLESOME ELEMENTS
					showMasks( pane, { resizing: true });
				}

			,	drag: function (e, ui) {
					if (!helperClassesSet) { // can only add classes after clone has been added to the DOM
						//$(".ui-draggable-dragging")
						ui.helper
							.addClass( helperClass +" "+ helperPaneClass ) // add helper classes
							.css({ right: "auto", bottom: "auto" })	// fix dir="rtl" issue
							.children().css("visibility","hidden")	// hide toggler inside dragged resizer-bar
						;
						helperClassesSet = true;
						// draggable bug!? RE-SET zIndex to prevent E/W resize-bar showing through N/S pane!
						if (s.isSliding) $Ps[pane].css("zIndex", z.pane_sliding);
					}
					// CONTAIN RESIZER-BAR TO RESIZING LIMITS
					var limit = 0;
					if (ui.position[side] < r.min) {
						ui.position[side] = r.min;
						limit = -1;
					}
					else if (ui.position[side] > r.max) {
						ui.position[side] = r.max;
						limit = 1;
					}
					// ADD/REMOVE dragging-limit CLASS
					if (limit) {
						ui.helper.addClass( helperLimitClass +" "+ helperPaneLimitClass ); // at dragging-limit
						window.defaultStatus = (limit>0 && pane.match(/(north|west)/)) || (limit<0 && pane.match(/(south|east)/)) ? o.tips.maxSizeWarning : o.tips.minSizeWarning;
					}
					else {
						ui.helper.removeClass( helperLimitClass +" "+ helperPaneLimitClass ); // not at dragging-limit
						window.defaultStatus = "";
					}
					// DYNAMICALLY RESIZE PANES IF OPTION ENABLED
					// won't trigger unless resizer has actually moved!
					if (live && Math.abs(ui.position[side] - lastPos) >= o.liveResizingTolerance) {
						lastPos = ui.position[side];
						resizePanes(e, ui, pane)
					}
				}

			,	stop: function (e, ui) {
					$('body').enableSelection(); // RE-ENABLE TEXT SELECTION
					window.defaultStatus = ""; // clear 'resizing limit' message from statusbar
					$R.removeClass( resizerClass +" "+ resizerPaneClass ); // remove drag classes from Resizer
					s.isResizing		= false;
					state.paneResizing	= false; // easy to see if ANY pane is resizing
					resizePanes(e, ui, pane, true); // true = resizingDone
				}

			});
		});

		/**
		* resizePanes
		*
		* Sub-routine called from stop() - and drag() if livePaneResizing
		*
		* @param {!Object}		evt
		* @param {!Object}		ui
		* @param {string}		pane
		* @param {boolean=}		[resizingDone=false]
		*/
		var resizePanes = function (evt, ui, pane, resizingDone) {
			var	dragPos	= ui.position
			,	c		= _c[pane]
			,	o		= options[pane]
			,	s		= state[pane]
			,	resizerPos
			;
			switch (pane) {
				case "north":	resizerPos = dragPos.top; break;
				case "west":	resizerPos = dragPos.left; break;
				case "south":	resizerPos = sC.layoutHeight - dragPos.top  - o.spacing_open; break;
				case "east":	resizerPos = sC.layoutWidth  - dragPos.left - o.spacing_open; break;
			};
			// remove container margin from resizer position to get the pane size
			var newSize = resizerPos - sC.inset[c.side];

			// Disable OR Resize Mask(s) created in drag.start
			if (!resizingDone) {
				// ensure we meet liveResizingTolerance criteria
				if (Math.abs(newSize - s.size) < o.liveResizingTolerance)
					return; // SKIP resize this time
				// resize the pane
				manualSizePane(pane, newSize, false, true); // true = noAnimation
				sizeMasks(); // resize all visible masks
			}
			else { // resizingDone
				// ondrag_end callback
				if (false !== _runCallbacks("ondrag_end", pane))
					manualSizePane(pane, newSize, false, true); // true = noAnimation
				hideMasks(true); // true = force hiding all masks even if one is 'sliding'
				if (s.isSliding) // RE-SHOW 'object-masks' so objects won't show through sliding pane
					showMasks( pane, { resizing: true });
			}
		};
	}

,	sizePane = function (evt_or_pane, size, skipCallback, noAnimation, force) {
		if (!isInitialized()) return;
		var	pane	= evtPane.call(this, evt_or_pane) // probably NEVER called from event?
		,	o		= options[pane]
		,	s		= state[pane]
		,	$P		= $Ps[pane]
		,	$R		= $Rs[pane]
		,	side	= _c[pane].side
		,	dimName	= _c[pane].sizeType.toLowerCase()
		,	skipResizeWhileDragging = s.isResizing && !o.triggerEventsDuringLiveResize
		,	doFX	= noAnimation !== true && o.animatePaneSizing
		,	oldSize, newSize
		;
		// QUEUE in case another action/animation is in progress
		$N.queue(function( queueNext ){
			// calculate 'current' min/max sizes
			setSizeLimits(pane); // update pane-state
			oldSize = s.size;
			size = _parseSize(pane, size); // handle percentages & auto
			size = max(size, _parseSize(pane, o.minSize));
			size = min(size, s.maxSize);
			if (size < s.minSize) { // not enough room for pane!
				queueNext(); // call before makePaneFit() because it needs the queue free
				makePaneFit(pane, false, skipCallback);	// will hide or close pane
				return;
			}

			// IF newSize is same as oldSize, then nothing to do - abort
			if (!force && size === oldSize)
				return queueNext();

			s.newSize = size;

			// onresize_start callback CANNOT cancel resizing because this would break the layout!
			if (!skipCallback && state.initialized && s.isVisible)
				_runCallbacks("onresize_start", pane);

			// resize the pane, and make sure its visible
			newSize = cssSize(pane, size);

			if (doFX && $P.is(":visible")) { // ANIMATE
				var fx		= $.layout.effects.size[pane] || $.layout.effects.size.all
				,	easing	= o.fxSettings_size.easing || fx.easing
				,	z		= options.zIndexes
				,	props	= {};
				props[ dimName ] = newSize +'px';
				s.isMoving = true;
				// overlay all elements during animation
				$P.css({ zIndex: z.pane_animate })
				  .show().animate( props, o.fxSpeed_size, easing, function(){
					// reset zIndex after animation
					$P.css({ zIndex: (s.isSliding ? z.pane_sliding : z.pane_normal) });
					s.isMoving = false;
					delete s.newSize;
					sizePane_2(); // continue
					queueNext();
				});
			}
			else { // no animation
				$P.css( dimName, newSize );	// resize pane
				delete s.newSize;
				// if pane is visible, then 
				if ($P.is(":visible"))
					sizePane_2(); // continue
				else {
					// pane is NOT VISIBLE, so just update state data...
					// when pane is *next opened*, it will have the new size
					s.size = size;				// update state.size
					$.extend(s, elDims($P));	// update state dimensions
				}
				queueNext();
			};

		});

		// SUBROUTINE
		function sizePane_2 () {
			/*	Panes are sometimes not sized precisely in some browsers!?
			 *	This code will resize the pane up to 3 times to nudge the pane to the correct size
			 */
			var	actual	= dimName==='width' ? $P.outerWidth() : $P.outerHeight()
			,	tries	= [{
						   	pane:		pane
						,	count:		1
						,	target:		size
						,	actual:		actual
						,	correct:	(size === actual)
						,	attempt:	size
						,	cssSize:	newSize
						}]
			,	lastTry = tries[0]
			,	thisTry	= {}
			,	msg		= 'Inaccurate size after resizing the '+ pane +'-pane.'
			;
			while ( !lastTry.correct ) {
				thisTry = { pane: pane, count: lastTry.count+1, target: size };

				if (lastTry.actual > size)
					thisTry.attempt = max(0, lastTry.attempt - (lastTry.actual - size));
				else // lastTry.actual < size
					thisTry.attempt = max(0, lastTry.attempt + (size - lastTry.actual));

				thisTry.cssSize = cssSize(pane, thisTry.attempt);
				$P.css( dimName, thisTry.cssSize );

				thisTry.actual	= dimName=='width' ? $P.outerWidth() : $P.outerHeight();
				thisTry.correct	= (size === thisTry.actual);

				// log attempts and alert the user of this *non-fatal error* (if showDebugMessages)
				if ( tries.length === 1) {
					_log(msg, false, true);
					_log(lastTry, false, true);
				}
				_log(thisTry, false, true);
				// after 4 tries, is as close as its gonna get!
				if (tries.length > 3) break;

				tries.push( thisTry );
				lastTry = tries[ tries.length - 1 ];
			}
			// END TESTING CODE

			// update pane-state dimensions
			s.size	= size;
			$.extend(s, elDims($P));

			if (s.isVisible && $P.is(":visible")) {
				// reposition the resizer-bar
				if ($R) $R.css( side, size + sC.inset[side] );
				// resize the content-div
				sizeContent(pane);
			}

			if (!skipCallback && !skipResizeWhileDragging && state.initialized && s.isVisible)
				_runCallbacks("onresize_end", pane);

			// resize all the adjacent panes, and adjust their toggler buttons
			// when skipCallback passed, it means the controlling method will handle 'other panes'
			if (!skipCallback) {
				// also no callback if live-resize is in progress and NOT triggerEventsDuringLiveResize
				if (!s.isSliding) sizeMidPanes(_c[pane].dir=="horz" ? "" : "center", skipResizeWhileDragging, force);
				sizeHandles();
			}

			// if opposite-pane was autoClosed, see if it can be autoOpened now
			var altPane = _c.oppositeEdge[pane];
			if (size < oldSize && state[ altPane ].noRoom) {
				setSizeLimits( altPane );
				makePaneFit( altPane, false, skipCallback );
			}

			// DEBUG - ALERT user/developer so they know there was a sizing problem
			if (tries.length > 1)
				_log(msg +'\nSee the Error Console for details.', true, true);
		}
	}

,	enableResizable = function (evt_or_pane) {
		if (!isInitialized()) return;
		var	pane = evtPane.call(this, evt_or_pane)
		,	$R	= $Rs[pane]
		,	o	= options[pane]
		;
		if (!$R || !$R.data('draggable')) return;
		o.resizable = true; 
		$R.draggable("enable");
		if (!state[pane].isClosed)
			$R	.css("cursor", o.resizerCursor)
			 	.attr("title", o.tips.Resize);
	}
	/**
	* @param {(string|Object)}	evt_or_pane
	*/
,	disableResizable = function (evt_or_pane) {
		if (!isInitialized()) return;
		var	pane = evtPane.call(this, evt_or_pane)
		,	$R	= $Rs[pane]
		;
		if (!$R || !$R.data('draggable')) return;
		options[pane].resizable = false; 
		$R	.draggable("disable")
			.css("cursor", "default")
			.attr("title", "");
		removeHover(null, $R[0]); // in case currently hovered
	}

	/**
	* Move a pane from source-side (eg, west) to target-side (eg, east)
	* If pane exists on target-side, move that to source-side, ie, 'swap' the panes
	*
	* @param {(string|Object)}	evt_or_pane1	The pane/edge being swapped
	* @param {string}			pane2			ditto
	*/
,	swapPanes = function (evt_or_pane1, pane2) {
		if (!isInitialized()) return;
		var pane1 = evtPane.call(this, evt_or_pane1);
		// change state.edge NOW so callbacks can know where pane is headed...
		state[pane1].edge = pane2;
		state[pane2].edge = pane1;
		// run these even if NOT state.initialized
		if (false === _runCallbacks("onswap_start", pane1)
		 ||	false === _runCallbacks("onswap_start", pane2)
		) {
			state[pane1].edge = pane1; // reset
			state[pane2].edge = pane2;
			return;
		}

		var
			oPane1	= copy( pane1 )
		,	oPane2	= copy( pane2 )
		,	sizes	= {}
		;
		sizes[pane1] = oPane1 ? oPane1.state.size : 0;
		sizes[pane2] = oPane2 ? oPane2.state.size : 0;

		// clear pointers & state
		$Ps[pane1] = false; 
		$Ps[pane2] = false;
		state[pane1] = {};
		state[pane2] = {};
		
		// ALWAYS remove the resizer & toggler elements
		if ($Ts[pane1]) $Ts[pane1].remove();
		if ($Ts[pane2]) $Ts[pane2].remove();
		if ($Rs[pane1]) $Rs[pane1].remove();
		if ($Rs[pane2]) $Rs[pane2].remove();
		$Rs[pane1] = $Rs[pane2] = $Ts[pane1] = $Ts[pane2] = false;

		// transfer element pointers and data to NEW Layout keys
		move( oPane1, pane2 );
		move( oPane2, pane1 );

		// cleanup objects
		oPane1 = oPane2 = sizes = null;

		// make panes 'visible' again
		if ($Ps[pane1]) $Ps[pane1].css(_c.visible);
		if ($Ps[pane2]) $Ps[pane2].css(_c.visible);

		// fix any size discrepancies caused by swap
		resizeAll();

		// run these even if NOT state.initialized
		_runCallbacks("onswap_end", pane1);
		_runCallbacks("onswap_end", pane2);

		return;

		function copy (n) { // n = pane
			var
				$P	= $Ps[n]
			,	$C	= $Cs[n]
			;
			return !$P ? false : {
				pane:		n
			,	P:			$P ? $P[0] : false
			,	C:			$C ? $C[0] : false
			,	state:		$.extend(true, {}, state[n])
			,	options:	$.extend(true, {}, options[n])
			}
		};

		function move (oPane, pane) {
			if (!oPane) return;
			var
				P		= oPane.P
			,	C		= oPane.C
			,	oldPane = oPane.pane
			,	c		= _c[pane]
			//	save pane-options that should be retained
			,	s		= $.extend(true, {}, state[pane])
			,	o		= options[pane]
			//	RETAIN side-specific FX Settings - more below
			,	fx		= { resizerCursor: o.resizerCursor }
			,	re, size, pos
			;
			$.each("fxName,fxSpeed,fxSettings".split(","), function (i, k) {
				fx[k +"_open"]  = o[k +"_open"];
				fx[k +"_close"] = o[k +"_close"];
				fx[k +"_size"]  = o[k +"_size"];
			});

			// update object pointers and attributes
			$Ps[pane] = $(P)
				.data({
					layoutPane:		Instance[pane]	// NEW pointer to pane-alias-object
				,	layoutEdge:		pane
				})
				.css(_c.hidden)
				.css(c.cssReq)
			;
			$Cs[pane] = C ? $(C) : false;

			// set options and state
			options[pane]	= $.extend(true, {}, oPane.options, fx);
			state[pane]		= $.extend(true, {}, oPane.state);

			// change classNames on the pane, eg: ui-layout-pane-east ==> ui-layout-pane-west
			re = new RegExp(o.paneClass +"-"+ oldPane, "g");
			P.className = P.className.replace(re, o.paneClass +"-"+ pane);

			// ALWAYS regenerate the resizer & toggler elements
			initHandles(pane); // create the required resizer & toggler

			// if moving to different orientation, then keep 'target' pane size
			if (c.dir != _c[oldPane].dir) {
				size = sizes[pane] || 0;
				setSizeLimits(pane); // update pane-state
				size = max(size, state[pane].minSize);
				// use manualSizePane to disable autoResize - not useful after panes are swapped
				manualSizePane(pane, size, true, true); // true/true = skipCallback/noAnimation
			}
			else // move the resizer here
				$Rs[pane].css(c.side, sC.inset[c.side] + (state[pane].isVisible ? getPaneSize(pane) : 0));


			// ADD CLASSNAMES & SLIDE-BINDINGS
			if (oPane.state.isVisible && !s.isVisible)
				setAsOpen(pane, true); // true = skipCallback
			else {
				setAsClosed(pane);
				bindStartSlidingEvents(pane, true); // will enable events IF option is set
			}

			// DESTROY the object
			oPane = null;
		};
	}

;	// END var DECLARATIONS

}

})( jQuery );
// END Layout - keep internal vars internal!
