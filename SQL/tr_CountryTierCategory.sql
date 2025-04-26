IF  EXISTS (SELECT * FROM sys.triggers WHERE object_id = OBJECT_ID(N'[dbo].[tr_CountryTierCategory]'))
DROP TRIGGER [dbo].[tr_CountryTierCategory]
GO
SET ANSI_NULLS ON
GO

SET QUOTED_IDENTIFIER ON
GO


CREATE TRIGGER [dbo].[tr_CountryTierCategory] ON [dbo].[CountryTierCategory] 
FOR UPDATE,DELETE
AS
/*********************************************************************/
/*                                                                   */
/*********************************************************************/
SET NOCOUNT ON

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
Declare		@Info			Varchar(25),
			@SeqNo			Int,
			@Max			Int,
			@Min			Int,
			@iID			Int,
			@iCountryCode	Char(5),
			@iTierId		Int,
			@iCategoryCode	Varchar(10),
			@iCategoryValue	Varchar(25),
			@iActive		Bit,
			@iLastUser		Int,
			@iLastUpdate	Char(10),
			@cCountryCode	Char(5),
			@cTierId		Int,
			@cCategoryCode	Varchar(10),
			@cCategoryValue	Varchar(25),
			@cActive		Bit,
			@cLastUser		Int,
			@cLastUpdate	Char(10)

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
If	(Select	Count(CTC.CountryCode) 
	From CountryTierCategory CTC INNER JOIN deleted DEL 
	ON CTC.CountryCode = DEL.CountryCode) > 0

	Begin
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Set	@Info =  'Update'
		
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Create Table #Temp
		(
			SeqNo		Int		Identity,
			CountryCode	Char(5)	NOT NULL,
			CategoryCode Varchar(10) NOT NULL
		)

		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Insert	#Temp
		Select	CountryCode,
				CategoryCode
		From	deleted
		
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Select	@Max	= Count(CountryCode),
				@SeqNo	= 1,
				@Min	= 1
		From	#Temp
		
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		WHILE @SeqNo <= @Max
			 Begin		-- BEGIN WHILE LOOP		
		
			/*********************************************************/
			/*                                                       */
			/*********************************************************/
			Select	@iCountryCode		= D.CountryCode,
					@iTierId			= D.TierId,
					@iCategoryCode		= D.CategoryCode,
					@iCategoryValue		= D.CategoryValue,
					@iActive			= D.Active,
					@iLastUser			= CONVERT(Char(10),D.LastUser,101),
					@iLastUpdate		= CONVERT(Char(10),D.LastUpdate,101)
			From	CountryTierCategory C INNER JOIN deleted D 
					ON C.CountryCode = D.CountryCode
			Where	D.CountryCode IN (Select CountryCode From #Temp Where SeqNo = @SeqNo)
			And		D.CategoryCode IN (Select CategoryCode From #Temp Where SeqNo = @SeqNo)
			
			/*********************************************************/
			/*                                                       */
			/*********************************************************/
			Select	@cCountryCode		= C.CountryCode,
					@cTierId			= C.TierId,
					@cCategoryCode		= C.CategoryCode,
					@cCategoryValue		= C.CategoryValue,
					@cActive			= C.Active,
					@cLastUser			= CONVERT(Char(10),C.LastUser,101),
					@cLastUpdate		= CONVERT(Char(10),C.LastUpdate,101)
			From	CountryTierCategory C INNER JOIN deleted D 
					ON C.CountryCode = D.CountryCode
			Where	C.CountryCode IN (Select CountryCode From #Temp Where SeqNo = @SeqNo) 
			And		C.CategoryCode IN (Select CategoryCode From #Temp Where SeqNo = @SeqNo)
			
			/*********************************************************/
			/*                                                       */
			/*********************************************************/
			IF
			 (
			 (@cTierId = @iTierId) And ----(@cCategoryCode = @iCategoryCode) And
			 (@cCategoryValue = @iCategoryValue) And (@cActive = @iActive) And (@cLastUser = @iLastUser)----- And
			 -----(@cLastUpdate = @iLastUpdate)
			 )
			 Begin			 
				/*****************************************************/
				/*					No Change(s)                     */
				/*****************************************************/
				GOTO NextRecordCheck
		 
			 End
			Else
			 Begin
				/*****************************************************/
				/*				Update Change(s)                     */
				/*****************************************************/
				Insert	AuditCountryTierCategory
				SELECT	D.CountryCode,
						D.TierId,
						D.CategoryCode,
						D.CategoryValue,
						D.Active,
						D.LastUser,
						D.LastUpdate,
						@Info,
						GETDATE(),
						USER_NAME()
				From	CountryTierCategory C INNER JOIN deleted D 
						ON C.CountryCode = D.CountryCode
				Where	D.CountryCode IN (Select CountryCode From #Temp Where SeqNo = @SeqNo)  
				And		D.CategoryCode IN (Select CategoryCode From #Temp Where SeqNo = @SeqNo)
									 
			 End
				/*****************************************************/
				/*                                                   */
				/*****************************************************/
				NextRecordCheck:	
				Select	@SeqNo = @SeqNo + @Min

			 End		-- END WHILE LOOP
			/*********************************************************/
			/*                                                       */
			/*********************************************************/
			Drop Table #Temp
		
	End
 Else
	Begin	
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Set	@Info =  'Delete'
		
		/*************************************************************/
		/*                                                           */
		/*************************************************************/
		Insert	AuditCountryTierCategory
		SELECT	CountryCode,
				TierId,
				CategoryCode,
				CategoryValue,
				Active,
				LastUser,
				LastUpdate,
				@Info,
				GETDATE(),
				USER_NAME()
		From	Deleted
		
	End

/*********************************************************************/
/*                                                                   */
/*********************************************************************/
SET NOCOUNT OFF
GO


